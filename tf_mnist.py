#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
import os
import numpy as np
import argparse
import shutil
from tensorflow.examples.tutorials.mnist import input_data

parser = argparse.ArgumentParser('MNIST Softmax')
parser.add_argument('--data_dir', type=str, default='/tmp/mnist-data', 
                                help='the directory of MNIST dataset')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--max_train_step', type=int, default=50000, help='the maximum training step')
parser.add_argument('--model_path', type=str, default='', help='the path of checkpoint file')
args = parser.parse_args()

def model():
    x = tf.placeholder(tf.float32, [None, 784], name='x')
    gt = tf.placeholder(tf.float32, [None, 10], name='groundtruth')
    with tf.variable_scope('layer1'):
        w1 = tf.get_variable('weight1', [784, 1024], initializer=tf.random_normal_initializer())
        b1 = tf.get_variable('bias1', [1024], initializer=tf.constant_initializer(0.0))
        h1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    with tf.variable_scope('layer2'):
        w2 = tf.get_variable('weight2', [1024, 1024], initializer=tf.random_normal_initializer())
        b2 = tf.get_variable('bias2', [1024], initializer=tf.constant_initializer(0.0))
        h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
    with tf.variable_scope('layer3'):
        w3 = tf.get_variable('weight3', [1024, 10], initializer=tf.random_normal_initializer())
        b3 = tf.get_variable('bias3', [10], initializer=tf.constant_initializer(0.0))
        y = tf.matmul(h2, w3) + b3
    # losses
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gt, logits=y))
    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(args.lr)
    # define one-step train ops
    train_op = optimizer.minimize(cross_entropy)
    return x, y, gt, train_op    
    
if __name__ == "__main__":
    max_train_step = args.max_train_step
    batch_size = args.batch_size
    mnist = input_data.read_data_sets(args.data_dir, one_hot=True)
    x, y, gt, train_op = model()
    
    # create saver
    saver = tf.train.Saver()
    if os.path.exists('./mnist'):
        print('=> directory is existed!')
    else:
        print('=> creating temporary directory ...')
        os.makedirs('./mnist')

    with tf.Session() as sess:
        if args.model_path == '':
            tf.global_variables_initializer().run()
        else:
            saver.restore(sess, args.model_path)

        for i in range(max_train_step):
            batch_x, batch_gt = mnist.train.next_batch(batch_size)
            sess.run(train_op, feed_dict={x: batch_x, gt: batch_gt})

            if i % 100 == 0:
                correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(gt, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                print('=> accuracy: {}'.format(sess.run(accuracy, feed_dict={x: mnist.test.images, gt: mnist.test.labels})))
                saver.save(sess, 'mnist/mnist_{:02d}.ckpt'.format(int(i / 100) + 1))
