"""
Define the recurrent entity network model.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from functools import partial

import tensorflow as tf

from entity_networks.dynamic_memory_cell import DynamicMemoryCell
from entity_networks.model_ops import cyclic_learning_rate, \
                                      get_sequence_length, \
                                      count_parameters, \
                                      prelu

OPTIMIZER_SUMMARIES = [
    "learning_rate",
    "loss",
    "gradients",
    "gradient_norm",
]

def get_input_encoding(inputs, initializer=None, scope=None):
    """
    Implementation of the learned multiplicative mask from Section 2.1, Equation 1.
    This module is also described in [End-To-End Memory Networks](https://arxiv.org/abs/1502.01852)
    as Position Encoding (PE). The mask allows the ordering of words in a sentence to affect the
    encoding.
    """
    with tf.variable_scope(scope, 'Encoding', initializer=initializer):
        _, _, max_sentence_length, embedding_size = inputs.get_shape().as_list()
        positional_mask = tf.get_variable(
            name='positional_mask',
            shape=[max_sentence_length, embedding_size])
        encoded_input = tf.reduce_sum(inputs * positional_mask, axis=2)
        return encoded_input

def get_output_module(
        last_state,
        query_embedding, #new
        encoded_query,
        true_inputs, #new
        embedding_matrix, #new
        mode, #new
        num_blocks,
        vocab_size,
        activation=tf.nn.relu,
        initializer=None,
        scope=None):
    """
    Implementation of Section 2.3, Equation 6. This module is also described in more detail here:
    [End-To-End Memory Networks](https://arxiv.org/abs/1502.01852).
    """
    # print ('vocab~~~~~~~~~~~~~')
    # print (vocab_size)
    with tf.variable_scope(scope, 'Output', initializer=initializer):
        last_state = tf.stack(tf.split(last_state, num_blocks, axis=1), axis=1)
        _, _, embedding_size = last_state.get_shape().as_list()

        # Use the encoded_query to attend over memories
        # (hidden states of dynamic last_state cell blocks)
        attention = tf.reduce_sum(last_state * encoded_query, axis=2)

        # Subtract max for numerical stability (softmax is shift invariant)
        attention_max = tf.reduce_max(attention, axis=-1, keep_dims=True)
        attention = tf.nn.softmax(attention - attention_max)
        attention = tf.expand_dims(attention, axis=2)

        # Weight memories by attention vectors
        u = tf.reduce_sum(last_state * attention, axis=1)

        # R acts as the decoder matrix to convert from internal state to the output vocabulary size
        hidden_size = 100
        R = tf.get_variable('R', [embedding_size, hidden_size])
        H = tf.get_variable('H', [embedding_size, embedding_size])

        q = tf.squeeze(encoded_query, axis=1)
        y = tf.matmul(activation(q + tf.matmul(u, H)), R)

        #################
        # seq2seq_begin #
        #################

        # encoder
        # seq_input = query
        # print ('QUERY_EMBEDDING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        # print (query_embedding.shape)
        # print (query_embedding)
        batch_size, _, max_length, hidden_size = query_embedding.get_shape().as_list()
        batch_size = tf.shape(query_embedding)[0]
        query_embedding = tf.reshape(query_embedding, [-1, max_length, hidden_size])
        seq_length = tf.ones([batch_size], dtype=tf.int32) * max_length
        y_temp = y
        for t in range(max_length-1):
            y = tf.concat((y, y_temp), 1)
        y = tf.reshape(y, [max_length, -1, hidden_size])
        y = tf.transpose(y, [1, 0, 2])

        lstm_fw = tf.contrib.rnn.LSTMCell(hidden_size/2, forget_bias=1.0)
        lstm_bw = tf.contrib.rnn.LSTMCell(hidden_size/2, forget_bias=1.0)
        encoder_outputs, state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=lstm_fw, cell_bw=lstm_bw, dtype=tf.float32,
            sequence_length=seq_length, inputs=query_embedding, time_major=False)
        output_fw, output_bw = encoder_outputs
        state_fw, state_bw = state
        encoder_outputs = tf.concat([y, output_fw, output_bw], 2)
        # print (encoder_outputs.shape)
        encoder_state_c = tf.concat((state_fw.c, state_bw.c), 1)
        encoder_state_h = tf.concat((state_fw.h, state_bw.h), 1)
        encoder_state = tf.contrib.rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)

        # decoder
        BOS = tf.ones([batch_size, 1], dtype=tf.int64)
        BOS = tf.expand_dims(BOS, 1)
        true_inputs = tf.concat([BOS, true_inputs], axis=-1)
        # print (true_inputs.shape)
        decoder_inputs = tf.nn.embedding_lookup(embedding_matrix, true_inputs)
        # print ('embedding_matrix~~~~~~~~~~~~')
        # print (embedding_matrix.shape)
        _, _, l, h = decoder_inputs.get_shape().as_list()
        decoder_inputs = tf.reshape(decoder_inputs, [-1, l, h])
        # print (decoder_inputs.shape)
        decoder_inputs = tf.unstack(decoder_inputs, axis=1)
        # print (decoder_inputs)
        # print ('~~~~~~~~~~~~~~~~~~~')
        
        def test_loop(prev, i):
            prev_index = tf.stop_gradient(tf.argmax(prev, axis=-1))
            pred_prev = tf.nn.embedding_lookup(embedding_matrix, prev_index)
            pred_prev = tf.reshape(pred_prev, [-1 ,h])
            return pred_prev
        def train_loop(prev, i):
            pred_prev = tf.nn.embedding_lookup(embedding_matrix, true_inputs[:, :, i])
            pred_prev = tf.reshape(pred_prev, [-1 ,h])
            return pred_prev

        cell = tf.contrib.rnn.LSTMCell(num_units=hidden_size)
        if mode == tf.contrib.learn.ModeKeys.TRAIN:
        	loop_function = train_loop
        elif mode == tf.contrib.learn.ModeKeys.INFER:
        	loop_function = test_loop

        output, _ = tf.contrib.legacy_seq2seq.attention_decoder(
					decoder_inputs = decoder_inputs,
					initial_state = encoder_state,
					attention_states = encoder_outputs,
					cell = cell,
					output_size = vocab_size,
					loop_function = loop_function, 
					)
        return output
        #################
        #  seq2seq_end  #
        #################

    outputs = None
    return outputs

def get_outputs(inputs, answers, params, mode):
    "Return the outputs from the model which will be used in the loss function."
    embedding_size = params['embedding_size']
    num_blocks = params['num_blocks']
    vocab_size = params['vocab_size']

    story = inputs['story']
    query = inputs['query']

    batch_size = tf.shape(story)[0]

    normal_initializer = tf.random_normal_initializer(stddev=0.1)
    ones_initializer = tf.constant_initializer(1.0)

    # Extend the vocab to include keys for the dynamic memory cell,
    # allowing the initialization of the memory to be learned.
    vocab_size = vocab_size + num_blocks

    with tf.variable_scope('EntityNetwork', initializer=normal_initializer):
        # PReLU activations have their alpha parameters initialized to 1
        # so they may be identity before training.
        alpha = tf.get_variable(
            name='alpha',
            shape=embedding_size,
            initializer=ones_initializer)
        activation = partial(prelu, alpha=alpha)

        # Embeddings
        embedding_params = tf.get_variable(
            name='embedding_params',
            shape=[vocab_size, embedding_size])

        # The embedding mask forces the special "pad" embedding to zeros.
        embedding_mask = tf.constant(
            value=[0 if i == 0 else 1 for i in range(vocab_size)],
            shape=[vocab_size, 1],
            dtype=tf.float32)
        embedding_params_masked = embedding_params * embedding_mask
        # print ('embedding_params_masked~~~~~~~~~~~~~~~~~~~~~~')
        # print (embedding_params_masked.shape)

        story_embedding = tf.nn.embedding_lookup(embedding_params_masked, story)
        query_embedding = tf.nn.embedding_lookup(embedding_params_masked, query)

        # Input Module
        encoded_story = get_input_encoding(
            inputs=story_embedding,
            initializer=ones_initializer,
            scope='StoryEncoding')
        encoded_query = get_input_encoding(
            inputs=query_embedding,
            initializer=ones_initializer,
            scope='QueryEncoding')

        # Memory Module
        # We define the keys outside of the cell so they may be used for memory initialization.
        # Keys are initialized to a range outside of the main vocab.
        keys = [key for key in range(vocab_size - num_blocks, vocab_size)]
        keys = tf.nn.embedding_lookup(embedding_params_masked, keys)
        keys = tf.split(keys, num_blocks, axis=0)
        keys = [tf.squeeze(key, axis=0) for key in keys]

        cell = DynamicMemoryCell(
            num_blocks=num_blocks,
            num_units_per_block=embedding_size,
            keys=keys,
            initializer=normal_initializer,
            recurrent_initializer=normal_initializer,
            activation=activation)

        # Recurrence
        initial_state = cell.zero_state(batch_size, tf.float32)
        sequence_length = get_sequence_length(encoded_story)
        _, last_state = tf.nn.dynamic_rnn(
            cell=cell,
            inputs=encoded_story,
            sequence_length=sequence_length,
            initial_state=initial_state)

        # Output Module
        outputs = get_output_module(
            last_state=last_state,
            encoded_query=encoded_query,
            query_embedding=query_embedding,
            true_inputs=answers, #new
            embedding_matrix=embedding_params_masked, #new
            mode=mode, #new
            num_blocks=num_blocks,
            vocab_size=vocab_size,
            initializer=normal_initializer,
            activation=activation)

        parameters = count_parameters()
        print('Parameters: {}'.format(parameters))

        return outputs

def get_predictions(outputs):
    "Return the actual predictions for use with evaluation metrics or TF Serving."
    outputs = tf.stack(outputs, axis=1)
    predictions = tf.cast(tf.argmax(outputs, axis=-1), tf.int32)
    return predictions

def get_loss(outputs, labels, labels_lengths, mode):
    "Return the loss function which will be used with an optimizer."

    loss = None
    if mode == tf.contrib.learn.ModeKeys.INFER:
        return loss
    batch_size = tf.shape(labels)[0]
    PAD = tf.ones([batch_size, 1], dtype=tf.int64)*0
    _, _, l = labels.get_shape().as_list()
    labels = tf.reshape(labels, [-1, l])
    targets = tf.concat([labels, PAD], axis=1)
    targets = tf.unstack(targets, axis=1)
    # print ('targets~~~~~~~~~~~~~~')
    # print (targets.shape)
    _, _, l = labels_lengths.get_shape().as_list()
    labels_lengths = tf.reshape(labels_lengths, [-1, l])
    # print ('label_length~~~~~~~~~~~~')
    # print (labels_lengths.shape)
    weights = tf.cast(labels_lengths, tf.float32)
    weights = tf.unstack(weights, axis=1)
    # loss = tf.contrib.seq2seq.sequence_loss(logits=tf.stack(outputs), targets=targets, weights=weights)
    loss = tf.contrib.legacy_seq2seq.sequence_loss(logits=outputs, targets=targets, weights=weights)

    return loss

def get_train_op(loss, params, mode):
    "Return the trainining operation which will be used to train the model."

    train_op = None
    if mode != tf.contrib.learn.ModeKeys.TRAIN:
        return train_op

    global_step = tf.contrib.framework.get_or_create_global_step()

    learning_rate = cyclic_learning_rate(
        learning_rate_min=params['learning_rate_min'],
        learning_rate_max=params['learning_rate_max'],
        step_size=params['learning_rate_step_size'],
        mode='triangular',
        global_step=global_step)
    tf.summary.scalar('learning_rate', learning_rate)

    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=global_step,
        learning_rate=learning_rate,
        optimizer='Adam',
        clip_gradients=params['clip_gradients'],
        gradient_noise_scale=params['gradient_noise_scale'],
        summaries=OPTIMIZER_SUMMARIES)

    return train_op

def model_fn(features, labels, mode, params):
    "Return ModelFnOps for use with Estimator."

    outputs = get_outputs(features, labels['answer'], params, mode)
    predictions = get_predictions(outputs)
    loss = get_loss(outputs, labels['answer'], labels['answer_length'], mode)
    # print ('loss~~~~~~~~~~~~~')
    # print (loss.shape)
    train_op = get_train_op(loss, params, mode)

    return tf.contrib.learn.ModelFnOps(
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        mode=mode)
