from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import json
import random
import argparse
import tensorflow as tf

from tqdm import tqdm

from entity_networks.inputs import generate_input_fn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dir',
        help='Directory containing data',
        default='data/ASUS/records/')
    args = parser.parse_args()

    metadata_path = os.path.join(args.data_dir, 'asus_169k_maxL50.json')
    with open(metadata_path) as metadata_file:
        metadata = json.load(metadata_file)

    filename = os.path.join(data_dir, 'asus_169k_maxL50_test.tfrecords')
    input_fn = generate_input_fn(
        filename=eval_filename,
        metadata=metadata,
        batch_size=BATCH_SIZE,
        num_epochs=1,
        shuffle=False)

    with tf.Graph().as_default():
        features, answers = input_fn()

        story = features['story']
        query = features['query']
        answer = answers['answer']
        answer_length = answers['answer_length']

        instances = []

        with tf.train.SingularMonitoredSession() as sess:
            while not sess.should_stop():
                story_, query_, answer_, answer_length_ = sess.run([story, query, answer, answer_length])

                instance = {
                    'story': story_[0].tolist(),
                    'query': query_[0].tolist(),
                    'answer': answer_[0].tolist(),
                    'answer_length': answer_length_[0].tolist(),
                }

                instances.append(instance)

        metadata['instances'] = random.sample(instances, k=10)

        output_path = 'asus_output.json'
        with open(output_path, 'w') as f:
            f.write(json.dumps(metadata))

if __name__ == '__main__':
    main()
