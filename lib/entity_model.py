from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from functools import partial

import tensorflow as tf
from lib import dynamic_memory_cell
from lib import model_ops
#from dynamic_memory_cell import DynamicMemoryCell
#from model_ops import cyclic_learning_rate, get_sequence_length, count_parameters, prelu

def get_input_encoding(inputs, initializer=None, scope=None):
    with tf.variable_scope(scope, 'Encoding', initializer=initializer):
        _, _, max_sentence_length, embedding_size = inputs.get_shape().as_list()
        positional_mask = tf.get_variable(
            name='positional_mask',
            shape=[max_sentence_length, embedding_size])
        encoded_input = tf.reduce_sum(inputs * positional_mask, axis=2)
        return encoded_input


def get_output_module( last_state, encoded_query, num_blocks, vocab_size, activation=tf.nn.relu, initializer=None, scope=None):

    with tf.variable_scope(scope, 'Output', initializer=initializer):
        last_state = tf.stack(tf.split(last_state, num_blocks, axis=1), axis=1)
        _, _, embedding_size = last_state.get_shape().as_list()

        # Use the encoded_query to attend over memories
        # (hidden states of dynamic last_state cell blocks)
        attention = tf.reduce_sum(last_state * encoded_query, axis=2)

        # Subtract max for numerical stability (softmax is shift invariant)
        attention_max = tf.reduce_max(attention, axis=-1, keep_dims=True)
        attention = tf.nn.softmax(attention - attention_max)
        new_attention = attention
        attention = tf.expand_dims(attention, axis=2)

        # Weight memories by attention vectors
        u = tf.reduce_sum(last_state * attention, axis=1)

        # R acts as the decoder matrix to convert from internal state to the output vocabulary size
        R = tf.get_variable('R', [embedding_size, vocab_size])
        H = tf.get_variable('H', [embedding_size, embedding_size])

        q = tf.squeeze(encoded_query, axis=1)
        y = tf.matmul(activation(q + tf.matmul(u, H)), R)
        return y, new_attention
    outputs = None
    attention = None
    return outputs, new_attention

def get_outputs(story, query, keys, params, embedding_matrix, buckets):
    "Return the outputs from the model which will be used in the loss function."
    num_blocks = params['key_number']
    vocab_size = params['vocab_size']
    embedding_size = params['embedding_size']
    batch_size = tf.shape(story)[0]
    size = embedding_size

    query = tf.stack(query, axis=1)
    query_temp = tf.expand_dims(query, 1)

    normal_initializer = tf.random_normal_initializer(stddev=0.1)
    ones_initializer = tf.constant_initializer(1.0)

    with tf.variable_scope('EntityNetwork', initializer=normal_initializer):
        # PReLU activations have their alpha parameters initialized to 1
        # so they may be identity before training.
        alpha = tf.get_variable(
            name='alpha',
            shape=embedding_size,
            initializer=ones_initializer)
        activation = partial(model_ops.prelu, alpha=alpha)

        story_embedding = tf.nn.embedding_lookup(embedding_matrix, story)
        query_embedding = tf.nn.embedding_lookup(embedding_matrix, query_temp)

        # Input Module
        encoded_story = get_input_encoding(
            inputs=story_embedding,
            initializer=ones_initializer,
            scope='StoryEncoding')
        encoded_query = get_input_encoding(
            inputs=query_embedding,
            initializer=ones_initializer,
            scope='QueryEncoding')
        
        # keys = [key for key in range(vocab_size - num_blocks, vocab_size)]
        keys = tf.nn.embedding_lookup(embedding_matrix, keys)
        keys = tf.split(keys, num_blocks, axis=0)
        keys = [tf.squeeze(key, axis=0) for key in keys]
		
        cell = dynamic_memory_cell.DynamicMemoryCell(
            num_blocks=num_blocks,
            num_units_per_block=embedding_size,
            keys=keys,
            initializer=normal_initializer,
            recurrent_initializer=normal_initializer,
            activation=activation)

        # Recurrence
        initial_state = cell.zero_state(batch_size, tf.float32)
        sequence_length = model_ops.get_sequence_length(encoded_story)
        _, last_state = tf.nn.dynamic_rnn(
            cell=cell,
            inputs=encoded_story,
            sequence_length=sequence_length,
            initial_state=initial_state)

        # Output Module
        outputs, attention = get_output_module(
            last_state=last_state,
            encoded_query=encoded_query,
            num_blocks=num_blocks,
            vocab_size=size,
            initializer=normal_initializer,
            activation=activation)

        parameters = model_ops.count_parameters()
        print('Parameters: {}'.format(parameters))

        return outputs, attention
