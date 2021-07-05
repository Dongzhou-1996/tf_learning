#!/usr/bin/env python
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import shutil
import os
from tensorflow.examples.tutorials.mnist import input_data

def model(lr=0.01):
    global_step = tf.Variable([], trainable=False, name='step')
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x')
        with tf.name_scope('input_visualize'):
            image_x = tf.reshape(x, [-1, 28, 28, 1])
            tf.summary.image('input image', image_x, 10)
        gt = tf.placeholder(tf.float32, [None, 10], name='groundtruth')
    with tf.name_scope('layer1'):
        with tf.name_scope('weight'):
            w1 = tf.Variable(tf.truncated_normal([784, 1024], stddev=1.0/28))
            tf.summary.histogram('weight1', w1)
        with tf.name_scope('bias'):
            b1 = tf.Variable(tf.zeros([1024]))
        with tf.name_scope('output'):
            y1 = tf.nn.relu(tf.matmul(x, w1) + b1)
            with tf.name_scope('hidden_output_vis'):
                image_hid = tf.reshape(y1, [-1, 32, 32, 1])
                tf.summary.image('hidden_output', image_hid, 10)
    with tf.name_scope('layer2'):
        with tf.name_scope('weight'):
            w2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=1.0/28))
            tf.summary.histogram('weight2', w2)
        with tf.name_scope('bias'):
            b2 = tf.Variable(tf.zeros([10]))
        with tf.name_scope('output'):
            y2 = tf.nn.softmax(tf.matmul(y1, w2) + b2)
    with tf.name_scope('cross_entropy'):
        loss = -tf.reduce_sum(gt * tf.log(tf.clip_by_value(y2, 1e-10, 1.0)))
        tf.summary.scalar('loss', loss)
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(lr)
        train_ops = optimizer.minimize(loss)
    with tf.name_scope('metric'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y2, 1), tf.argmax(gt, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)
    return x, gt, train_ops, accuracy

if __name__ == '__main__':
    log_dir = './tensorboard_log'
    if os.path.exists(log_dir):
        print('=> deleting tensorboard log directory ...')
        shutil.rmtree(log_dir)
        print('=> creating tensorboard log directory ...')
        os.makedirs(log_dir)

    max_train_steps = 5000
    batch_size = 64
    x, gt, train_ops, accuracy = model(lr=0.001)
    mnist = input_data.read_data_sets('/tmp/mnist', one_hot=True)
    merged = tf.summary.merge_all()
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(os.path.join(log_dir, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(log_dir, 'test'))
        tf.global_variables_initializer().run()
        
        for i in range(max_train_steps):
            # acquire data
            batch_x, batch_gt = mnist.train.next_batch(batch_size, fake_data=False)
            
            if i % 10 == 0:
                summary, acc = sess.run([merged, accuracy], {x: mnist.test.images, gt: mnist.test.labels})
                print('=> accuracy: {}'.format(acc))
                test_writer.add_summary(summary, i)
            
            if i % 100 == 99:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                _, summary = sess.run([train_ops, merged], 
                                  feed_dict={x: batch_x, gt: batch_gt},
                                  options=run_options, run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step{:03d}'.format(i))
                train_writer.add_summary(summary, i)
            else:
                _, summary = sess.run([train_ops, merged], feed_dict={x: batch_x, gt: batch_gt})
                train_writer.add_summary(summary, i)
            
    train_writer.close()
    test_writer.close()


