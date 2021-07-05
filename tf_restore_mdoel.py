#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
import os
import shutil


# create variables
x = tf.placeholder(dtype=tf.float32)
w = tf.Variable(0.0, name='weight')
b = tf.Variable(1.0, name='bias')
y = w * x + b

saver = tf.train.Saver()
with tf.Session() as sess:
    for i in range(4):
        model_path = './temp/test_{:02d}.ckpt'.format(i+1)
        print('=> load model params from checkpoint file ...')
        saver.restore(sess, model_path)

        for j in range(4):
            sess.run(tf.assign_sub(w, 1.0))
            sess.run(tf.assign_add(b, 1.0))
            print('=> w={}, b={}'.format(w.eval(), b.eval()))

