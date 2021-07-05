#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
import os
import shutil

if os.path.exists('./temp'):
    print('=> deleting old temporary directory ...')
    shutil.rmtree('./temp')
    print('=> creating new temporary directory ...')
    os.makedirs('./temp')
else:
    print('=> creating temporary directory ...')
    os.makedirs('./temp')

# create variables
x = tf.placeholder(dtype=tf.float32)
w = tf.Variable(0.0, name='weight')
b = tf.Variable(1.0, name='bias')
y = w*x + b

# create saver instance
saver = tf.train.Saver(var_list=tf.global_variables())
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(4):
        sess.run(tf.assign_add(w, 1.0))
        sess.run(tf.assign_sub(b, 0.001))
        print('w={}, b={}'.format(w.eval(), b.eval()))
        saver.save(sess, 'temp/test_{:02d}.ckpt'.format(i+1))
        print('=> checkpoint file have been successfuly written down')
