import os
import sys
import jieba
import tensorflow as tf

from lib import data_utils
from lib.seq2seq_model_utils import create_model, get_predicted_sentence
jieba.set_dictionary('dict.txt.big')

def chat(args):
  with tf.Session() as sess:
    # Create model and load parameters.
    args.batch_size = 1  # We decode one sentence at a time.
    model = create_model(sess, args)

    # Load vocabularies.
    vocab_path = os.path.join(args.data_dir, "vocab%d.in" % args.vocab_size)
    vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)

    # Decode from standard input.
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    jieba_sentence = ' '.join(jieba.cut(sentence.strip()))
    history_list = ['', '', '', '', jieba_sentence]

    while jieba_sentence:
        print (history_list)
        predicted_sentence = get_predicted_sentence(args, history_list[:4], history_list[4], vocab, rev_vocab, model, sess)
        # print(predicted_sentence)
        predicted = [] 
        if isinstance(predicted_sentence, list):
            for sent in predicted_sentence:
                print("  (%s) -> %s" % (sent['prob'], sent['dec_inp']))
                predicted = sent['dec_inp']
        else:
            print(jieba_sentence, ' -> ', predicted_sentence)
        history_list[0] = history_list[2]
        history_list[1] = history_list[3]
        history_list[2] = history_list[4]
        history_list[3] = predicted

        #new
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        jieba_sentence = ' '.join(jieba.cut(sentence.strip()))
        history_list[4] = jieba_sentence

