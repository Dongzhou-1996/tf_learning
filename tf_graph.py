#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf

g1 = tf.Graph()
with g1.as_default():
    # set graph 1 as default graph in this context
    a = tf.Variable(0.0, name='a')
    assert a.graph is g1
    print('=> variable a is belong to graph1')
with tf.Graph().as_default() as g2:
    # create graph2 and set it as default graph in this context
    b = tf.Variable(1.1, name='b')
    assert b.graph is g2
    print('=> variable b is belong to graph2')
