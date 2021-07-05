#!/usr/bin/env python
# coding=utf-8
from __future__ import print_function
import os
import glob
import warnings
import tensorflow as tf

warnings.filterwarnings('ignore')

# label length
LABEL_BYTES = 1
# image size
IMAGE_SIZE = 32
# image channel
IMAGE_CHANNEL = 3
# image data length
IMAGE_BYTES = IMAGE_SIZE * IMAGE_SIZE * IMAGE_CHANNEL
# classes num
NUM_CLASSES = 10


def read_cifar10(data_file, batch_size=16):
    """
    input params:
	data_file: CIFAR-10 data file
	batch_size: the size of batch
    returns:
	images: images batch following format [batch_size, IMAGE_SIZE, IMAGE_SIZE]
	labels: labels batch following format [batch_size, NUM_CLASSES]
    """
    sess = tf.InteractiveSession()
    record_bytes = LABEL_BYTES + IMAGE_BYTES
    # create filename list
    data_files = glob.glob(os.path.join(data_file, '*_batch*'))
    print(data_files)
    # create filename queue
    filename_queue = tf.train.string_input_producer(data_files, shuffle=True)
    # create reader for binary file
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    _, value = reader.read(filename_queue)
    print('value: {}'.format(value))
    # split examples to labels and images
    record = tf.reshape(tf.decode_raw(value, tf.uint8), [record_bytes])
    label = tf.cast(tf.slice(record, begin=[0], size=[LABEL_BYTES]), tf.int32)
    print('label: {}'.format(label))
    # transfer [depth * height * width] record to [depth, height, width] image tensor
    image = tf.reshape(tf.slice(record, [LABEL_BYTES], [IMAGE_BYTES]),
                       [IMAGE_CHANNEL, IMAGE_SIZE, IMAGE_SIZE])
    # transfer [depth, height, width] format image to [height, width, depth]
    image = tf.cast(tf.transpose(image, [1, 2, 0]), tf.float32)

    # create examples queue
    example_queue = tf.RandomShuffleQueue(capacity=16 * batch_size, 
                                         min_after_dequeue=8 * batch_size,
                                         dtypes=[tf.float32, tf.int32],
                                         shapes = [[IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL], [1]])
    #num_threads = 2
    # create examples enqueue op
    example_enqueue_op = example_queue.enqueue([image, label])
    # add multi-threads to QueueRunner
    #tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(
    #                                example_queue, [example_enqueue_op]*num_threads))
    
    # group batch images and labels
    images, labels = example_queue.dequeue_many(batch_size)
    labels = tf.reshape(labels, [batch_size, 1])
    print(sess.run([labels]))
    indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
    
    labels = tf.sparse_to_dense(tf.concat(values=[tf.zeros(labels.shape[0], 1), labels], axis=1),
                               [batch_size, NUM_CLASSES], 1.0, 0.0)


if __name__ == "__main__":
    cifar_path = '/home/dzhou/Dataset/cifar-10/'
    read_cifar10(cifar_path, batch_size=16)
