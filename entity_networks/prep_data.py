# -*- coding: utf-8 -*-
"""
Loads and pre-processes a ASUS dataset into TFRecords.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

# from imp import reload

import os
import re
import sys
import json
import jieba
import tarfile
import tensorflow as tf
import numpy as np

from tqdm import tqdm

# set sys encoding
# reload(sys)
# sys.setdefaultencoding("utf-8")

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'source_path',
    'data/華碩客服資料.txt',
    'Tar containing ASUS sources.')
tf.app.flags.DEFINE_string('output_dir', 'data/ASUS/records/', 'Dataset destination.')
tf.app.flags.DEFINE_boolean('only_50k', False, 'Whether to use ASUS 50k or ASUS 169k (default 169,175).')

SPLIT_RE = re.compile(r'(\W+)?')

PAD_TOKEN = '_PAD'
PAD_ID = 0
MAX_L = 50
train_test_split = 615218

def tokenize(sentence):
    "Tokenize a string by splitting on non-word characters and stripping whitespace."
    # return [token.strip().lower() for token in re.split(SPLIT_RE, sentence) if token.strip()]
    return [token.strip().lower() for token in jieba.cut(sentence) if token.strip()]

def parse_stories(lines, only_supporting=False):
    """
    Parse the ASUS task format.
    """
    stories = []
    story = []
    user = ''
    skip = False
    n_full_story = 0
    n_part_story = 0
    for line in lines:
        line = line.strip()
        if line == '':
            if skip:
                n_part_story += 1
            else:
                n_full_story += 1
            story = []
            skip = False
            user = 'Server'
            continue
        if skip:
            continue
        specify_user = False
        change_user = False
        if ':' in line:
            new_user, new_line = line.split(':', 1)
            if new_user == 'Server' or new_user == 'Client':
                change_user = (user != new_user)
                user = new_user
                line = new_line
                specify_user = True
        if len(story) == 0 and user == 'Server':
            continue
        if user == 'Server':
            answer = tokenize(line)
            if specify_user and change_user:
                query = story[-1][1:] 
                substory = [x for x in story[:-1] if x]
                story.append(['<' + user + '>'] + answer)
                stories.append([substory, query, answer + ['<EOS>']])
            else:
                story[-1] += answer
                stories[-1][-1] = story[-1][1:] + ['<EOS>']
            if len(story[-1]) > MAX_L-1:
                skip = True
                stories = stories[:-1]
                continue
        else:
            sentence = tokenize(line)
            if specify_user and change_user:
                story.append(['<' + user + '>'] + sentence)
            else:
                story[-1] += sentence
            if len(story[-1]) > MAX_L-1:
                skip = True
                continue
    print("~~~~~~~~Full stories: {0}".format(n_full_story))
    print("~~~~~~~~Part stories: {0}".format(n_part_story))
    print("~~~~~~~~Total: {0}".format(n_full_story + n_part_story))
    stories = [(x[0], x[1], x[2], len(x[2])+1) for x in stories]
    return stories[:train_test_split], stories[train_test_split:]

def save_dataset(stories, path):
    """
    Save the stories into TFRecords.

    NOTE: Since each sentence is a consistent length from padding, we use
    `tf.train.Example`, rather than a `tf.train.SequenceExample`, which is
    _slightly_ faster.
    """
    writer = tf.python_io.TFRecordWriter(path)
    counter = 0
    print("~~~~~~~~Ready to write~~~~~~~~")
    for story, query, answer, ans_length in stories:
        # print("~~~Flat story~~~", end='')
        story_flat = [token_id for sentence in story for token_id in sentence]

        # print("~~~Feature sep~~~", end='')
        story_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=story_flat))
        query_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=query))
        answer_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=answer))
        ans_length_weight = np.zeros(MAX_L, dtype=int)
        ans_length_weight[:ans_length] = np.ones(ans_length, dtype=int)
        ans_length_weight = ans_length_weight.tolist()
        ans_length_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=ans_length_weight))

        # print("~~~all~~~", end='')
        features = tf.train.Features(feature={
            'story': story_feature,
            'query': query_feature,
            'answer': answer_feature,
            'answer_length': ans_length_feature,
        })

        # print("~~~Example~~~", end='')
        example = tf.train.Example(features=features)
        # print("~~~Write file~~~")
        writer.write(example.SerializeToString())

        counter += 1
        print("{0}".format(counter), end='\r')
    print("~~~finish~~~")
    writer.close()

def tokenize_stories(stories, token_to_id):
    "Convert all tokens into their unique ids."
    story_ids = []
    for story, query, answer, ans_length in stories:
        story = [[token_to_id[token] for token in sentence] for sentence in story]
        query = [token_to_id[token] for token in query]
        answer = [token_to_id[token] for token in answer]
        story_ids.append((story, query, answer, ans_length))
    return story_ids

def get_tokenizer(stories):
    "Recover unique tokens as a vocab and map the tokens to ids."
    tokens_all = []
    for story, query, answer, _ in stories:
        tokens_all.extend([token for sentence in story for token in sentence[1:]] + query + answer)
    vocab = [PAD_TOKEN, '<BOS>', '<EOS>', '<Client>', '<Server>'] + sorted(set(tokens_all))
    token_to_id = {token: i for i, token in enumerate(vocab)}
    return vocab, token_to_id

def pad_stories(stories, max_sentence_length, max_story_length, max_query_length, max_answer_length):
    "Pad sentences, stories, and queries to a consistence length."
    for story, query, answer, ans_length in stories:
        for sentence in story:
            for _ in range(max_sentence_length - len(sentence)):
                sentence.append(PAD_ID)
            assert len(sentence) == max_sentence_length

        for _ in range(max_story_length - len(story)):
            story.append([PAD_ID for _ in range(max_sentence_length)])

        for _ in range(max_query_length - len(query)):
            query.append(PAD_ID)

        assert len(answer) == ans_length-1
        for _ in range(max_answer_length - len(answer)):
            answer.append(PAD_ID)

        assert len(story) == max_story_length
        assert len(query) == max_query_length
        assert len(answer) == max_answer_length

    print("~~~~~~~~Finish padding~~~~~~~~")
    sys.stdout.flush()
    return stories

def truncate_stories(stories, max_length):
    "Truncate a story to the specified maximum length."
    stories_truncated = []
    for story, query, answer, ans_length in stories:
        # if len(query) > MAX_L or len(answer) > MAX_L:
            # continue
        # if max([len(sentence) for sentence in story]) > MAX_L:
            # continue
        story_truncated = story[-max_length:]
        stories_truncated.append((story_truncated[:][:MAX_L-1], query, answer, ans_length))
    return stories_truncated

def main():
    "Main entrypoint."

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    task_names = [
        'asus_server_chatbot',
    ]

    task_titles = [
        'ASUS Server Chatbot',
    ]

    task_ids = [
        'asus',
    ]

    for task_id, task_name, task_title in tqdm(zip(task_ids, task_names, task_titles), \
            desc='Processing datasets into records...'):
        stories_path = os.path.join(FLAGS.source_path)
        if FLAGS.only_50k:
            dataset_path_train = os.path.join(FLAGS.output_dir, task_id + '_50k_maxL' + str(MAX_L) + '_train.tfrecords')
            dataset_path_test = os.path.join(FLAGS.output_dir, task_id + '_50k_maxL' + str(MAX_L) + '_test.tfrecords')
            metadata_path = os.path.join(FLAGS.output_dir, task_id + '_50k_maxL' + str(MAX_L) + '.json')
            task_size = 50000
        else:
            dataset_path_train = os.path.join(FLAGS.output_dir, task_id + '_169k_maxL' + str(MAX_L) + '_train.tfrecords')
            dataset_path_test = os.path.join(FLAGS.output_dir, task_id + '_169k_maxL' + str(MAX_L) + '_test.tfrecords')
            metadata_path = os.path.join(FLAGS.output_dir, task_id + '_169k_maxL' + str(MAX_L) + '.json')
            task_size = 169175

        # From the entity networks paper:
        # > Copying previous works (Sukhbaatar et al., 2015; Xiong et al., 2016),
        # > the capacity of the memory was limited to the most recent 70 sentences,
        # > except for task 3 which was limited to 130 sentences.
        if task_id == 'qa3':
            truncated_story_length = 130
        else:
            truncated_story_length = 70

        # tar = tarfile.open(FLAGS.source_path)

        f_story = open(stories_path, 'r')

        print("~~~~~Parse stories~~~~~")
        stories_train, stories_test = parse_stories(f_story.readlines())

        print("~~~~~Truncate stories~~~~~")
        stories_train = truncate_stories(stories_train, truncated_story_length)
        stories_test = truncate_stories(stories_test, truncated_story_length)

        print("~~~~~Get tokenizer~~~~~")
        vocab, token_to_id = get_tokenizer(stories_train + stories_test)
        vocab_size = len(vocab)

        print("~~~~~Tokenize stories~~~~~")
        stories_token_train = tokenize_stories(stories_train, token_to_id)
        stories_token_test = tokenize_stories(stories_test, token_to_id)
        stories_token_all = stories_token_train + stories_token_test
        del stories_train
        del stories_test
        del token_to_id

        print("~~~~~Max sentence length~~~~~", end='')
        story_lengths = [len(sentence) for story, _, _, _ in stories_token_all for sentence in story]
        max_sentence_length = max(story_lengths)
        print(str(max_sentence_length))
        print("~~~~~Average sentence length~~~~~{0}".format(sum(story_lengths)/float(len(story_lengths))))
        del story_lengths

        print("~~~~~Max story length~~~~~", end='')
        max_story_length = max([len(story) for story, _, _, _ in stories_token_all])
        print(str(max_story_length))

        print("~~~~~Max query length~~~~~", end='')
        max_query_length = max([len(query) for _, query, _, _ in stories_token_all])
        print(str(max_query_length))

        print("~~~~~Max answer length~~~~~", end='')
        max_answer_length = max([ans_length for _, _, _, ans_length in stories_token_all])
        print(str(max_answer_length))
        del stories_token_all

        print("~~~~~Output metadata file~~~~~")
        with open(metadata_path, 'w') as f:
            metadata = {
                'task_id': task_id,
                'task_name': task_name,
                'task_title': task_title,
                'task_size': task_size,
                'max_query_length': max_query_length,
                'max_story_length': max_story_length,
                'max_sentence_length': max_sentence_length,
                'max_answer_length': max_answer_length,
                'vocab': vocab,
                'vocab_size': vocab_size,
                'filenames': {
                    'train': os.path.basename(dataset_path_train),
                    'test': os.path.basename(dataset_path_test),
                }
            }
            json.dump(metadata, f)

        print("~~~~~Pad stories~~~~~")
        stories_pad_train = pad_stories(stories_token_train, \
            max_sentence_length, max_story_length, max_query_length, max_answer_length)
        stories_pad_test = pad_stories(stories_token_test, \
            max_sentence_length, max_story_length, max_query_length, max_answer_length)
        del stories_token_train
        del stories_token_test

        print("~~~~~Save dataset~~~~~")
        save_dataset(stories_pad_train, dataset_path_train)
        save_dataset(stories_pad_test, dataset_path_test)

if __name__ == '__main__':
    main()

