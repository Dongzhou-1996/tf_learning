#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
import numpy as np
with tf.name_scope("placeholderExample"):
    X = tf.placeholder(tf.float32, shape=(2, 2), name='X')
    # Y = tf.placeholder(tf.float32, shape=(2, 2), name='Y')
    Z = tf.matmul(X, X, name='matmul')
with tf.Session() as sess:
    x = np.random.rand(2, 2)
    #y = np.random.rand(2, 2)
    print(sess.run(Z, feed_dict={X: x})) 
