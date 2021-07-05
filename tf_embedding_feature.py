#!/usr/bin/env python
# coding=utf-8
import argparse
import sys
import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data

parser = argparse.ArgumentParser('embedding feature visualization')
parser.add_argument('--max_nums', type=int, default=10000, help='the number of steps to run trainner')
parser.add_argument('--data_dir', type=str, default='/tmp/mnist/input_data', help='the directory of dataset')
parser.add_argument('--log_dir', type=str, default='log/summary/', help='the direcoty of summary logs')
args = parser.parse_args()

def main(_):
    # create log direcotry
    if os.path.exists(args.log_dir):
        shutil.rmtree(args.log_dir)
    else:
        os.makedirs(args.log_dir)

    # mnist loader
    mnist = input_data.read_data_sets(args.data_dir, one_hot=True, fake_data=False)

    # create embedding variable
    embedding_var = tf.Variable(tf.stack(mnist.test.images[:args.max_nums]), trainable=False, name='embedding')

    # session
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # create embedding_saver
        embedding_saver = tf.train.Saver()
        embedding_saver.save(sess, os.path.join(args.log_dir, 'model.ckpt'))

        # create metadata file and wirte down labels to metadata file
        metadata_file = os.path.join(args.log_dir, 'metadata.tsv')
        with open(metadata_file, 'w') as f:
            for i in range(args.max_nums):
                c = np.nonzero(mnist.test.labels[::1])[1:][0][i]
                f.write('{}\n'.format(c))

        # create FileWriter to save graph
        writer = tf.summary.FileWriter(args.log_dir, sess.graph)
        # create projector config parameters
        config = projector.ProjectorConfig()
        embeddings = config.embeddings.add()
        embeddings.tensor_name = 'embedding:0'
        embeddings.metadata_path = 'metadata.tsv'
        embeddings.sprite.image_path = 'images/mnist_10k_sprite.png'
        embeddings.sprite.single_image_dim.extend([28, 28])
        projector.visualize_embeddings(writer, config)

if __name__ == "__main__":
    tf.app.run(main)

