#!/usr/bin/env python
# coding=utf-8
from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math
import sys
import tempfile
import time
import argparse

from tensorflow.examples.tutorials.mnist import input_data

parser = argparse.ArgumentParser('synchronize distributed training')
parser.add_argument('--data_dir', type=str, default='/tmp/mnist-data', help='the direcotry of mnist dataset')
parser.add_argument('--log_dir', type=str, default='log', help='the log directory of training')
parser.add_argument('--task_index', type=int, default=None, help='the index of task')
parser.add_argument('--replicas_to_aggregate', type=int, default=None, help='number of replicas to aggregate before parameter update is applied')
parser.add_argument('--sync_replicas', type=bool, default=None, help='use the sync_replicas mode')
parser.add_argument('--ps_host', type=str, default='localhost:1222', help='Comma-separated list of hostname:port pairs')
parser.add_argument('--worker_host', type=str, default='localhost:2222', help='Comma-separated list of hostname:port pairs')
parser.add_argument('--job_name', type=str, choices={'worker', 'PS'}, help='the name of job')
parser.add_argument('--num_gpus', type=int, default=None, help='the number of GPUs')
parser.add_argument('--hidden_units', type=int, default=256, help='the number of units in hidden layer of NN model')
parser.add_argument('--train_step', type=int, default=1000, help='the global steps of training')
parser.add_argument('--batch_size', type=int, default=64, help='the size of batch')
parser.add_argument('--lr', type=float, default=0.01, help='the learning rate')
args = parser.parse_args()

IMAGE_PIXEL = 28

def model(worker_device, cluster, ps_device='/job:PS/cpu:0', num_worker=None, 
          hidden_units=256, lr=0.01):
    with tf.device(tf.train.replica_device_setter(
        worker_device=worker_device, ps_device=ps_device, cluster=cluster)):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        x = tf.placeholder(tf.float32, [None, IMAGE_PIXEL*IMAGE_PIXEL])
        gt = tf.placeholder(tf.float32, [None, 10])
        with tf.variable_scope('layer1'):
            w1 = tf.Variable(tf.truncated_normal([IMAGE_PIXEL*IMAGE_PIXEL, hidden_units], 
                                                 stddev=1.0/IMAGE_PIXEL), name='weight1')
            b1 = tf.Variable(tf.zeros([hidden_units]), name='bias1')
            y1 = tf.nn.relu(tf.matmul(x, w1) + b1)
        with tf.variable_scope('softmax'):
            w2 = tf.Variable(tf.truncated_normal([hidden_units, 10], 
                                                  stddev=1.0/IMAGE_PIXEL), name='weight2')
            b2 = tf.Variable(tf.zeros([10]), name='bias2')
            y2 = tf.nn.softmax(tf.matmul(y1, w2) + b2)
        # define loss function
        cross_entropy = -tf.reduce_sum(gt * tf.log(tf.clip_by_value(y2, 1e-10, 1.0)))
        optimizer = tf.train.AdamOptimizer(lr)
    
    if args.sync_replicas:
        if args.replicas_to_aggregate is None:
            replicas_to_aggregate = num_worker
        else:
            replicas_to_aggregate = args.replicas_to_aggregate
        optimizer = tf.train.SyncReplicasOptimizer(optimizer,
                                                  replicas_to_aggregate=replicas_to_aggregate,
                                                  total_num_replicas=num_worker,
                                                  name='synchronize_optimizer')
    train_op = optimizer.minimize(cross_entropy, global_step=global_step)
    
    # calc accuracy
    correct_prediction = tf.equal(tf.argmax(y2, 1), tf.argmax(gt, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    is_chief = (args.task_index == 0)
    if args.sync_replicas:
        local_init_op = optimizer.local_step_init_op
        if is_chief:
            local_init_op = optimizer.chief_init_op
        ready_for_local_init_op = optimizer.ready_for_local_init_op
        chief_queue_runner = optimizer.get_chief_queue_runner()
        sync_init_op = optimizer.get_init_tokens_op()
    else:
        sync_init_op = None
    init_op = tf.global_variables_initializer()
    
    if args.sync_replicas:
        supervisor = tf.train.Supervisor(is_chief=is_chief,
                                        logdir=args.log_dir,
                                        init_op=init_op,
                                        local_init_op=local_init_op,
                                        ready_for_local_init_op=ready_for_local_init_op,
                                        recovery_wait_secs=1,
                                        global_step=global_step)
    else:
        supervisor = tf.train.Supervisor(is_chief=is_chief,
                                        logdir=args.log_dir,
                                        init_op=init_op,
                                        recovery_wait_secs=1,
                                        global_step=global_step)
    return sync_init_op, x, y2, gt, train_op, accuracy, global_step, supervisor, chief_queue_runner
    
def main(unused_argv):
    mnist = input_data.read_data_sets(args.data_dir, one_hot=True)
    if args.task_index is None or args.task_index == '':
        raise ValueError('=> must specify an explicit task index!')
    # parse the host name:port pairs of PS and workers
    ps_spec = args.ps_host.split(',')
    worker_spec = args.worker_host.split(',')
    # the number of workers
    num_worker = len(worker_spec)
    print('=> number of workers: {}'.format(num_worker))
    cluster = tf.train.ClusterSpec({'PS': ps_spec, 'worker': worker_spec})

    server = tf.train.Server(cluster, job_name=args.job_name, task_index=args.task_index)
    # if job_name is PS, start server immediately and listen requist from workers
    if args.job_name == 'PS':
        server.join()
    
    is_chief = (args.task_index == 0)

    if args.num_gpus > 0:
        if args.num_gpus < num_worker:
            raise ValueError('=> number of GPUs is less than number of workers!')
        gpu = args.task_index % args.num_gpus
        worker_device = '/job:worker/task:{}/gpu:{}'.format(args.task_index, gpu)
    elif args.num_gpus == 0:
        cpu = 0
        worker_device = '/job:worker/task:{}/cpu:{}'.format(args.task_index, cpu)

    sync_init_op, x, y, gt, train_op, accuracy, global_step, supervisor, chief_queue_runner = model(worker_device, cluster=cluster, num_worker=num_worker, hidden_units=args.hidden_units, lr=args.lr)
    
    config = tf.ConfigProto(allow_soft_placement=True,
                           device_filters=['/job:PS',
                                          '/job:worker/task:{}'.format(args.task_index)])
    if is_chief:
        print('=> worker {}: initializing session ...'.format(args.task_index))
    else:
        print('=> worker {}: waiting for session to be initialized in chief worker ...'.format(args.task_index))
    
    sess = supervisor.prepare_or_wait_for_session(server.target, config=config)
    print('=> worker {}: session initialization  complete.'.format(args.task_index))
    if args.sync_replicas and is_chief:
        # initialize synchronize token queue
        sess.run([sync_init_op])
        # start 3 threads with QueueRunner, and run their standard service
        supervisor.start_queue_runners(sess, [chief_queue_runner])

    time_start = time.time()
    print('=> training begins @ {}'.format(time_start))
    local_step = 0
    while True:
        batch_x, batch_gt = mnist.train.next_batch(args.batch_size)
        train_feed = {x: batch_x, gt: batch_gt}
        _, step = sess.run([train_op, global_step], feed_dict=train_feed)
        local_step += 1
        now = time.time()
        print('=> {}: worker {}: train step {} is done (global step: {})'.format(now, args.task_index, local_step, step))
        # validate 
        if local_step % 100 == 0:
            val_feed = {x: mnist.validation.images, gt: mnist.validation.labels}
            print('=> validate accuracy: {}'.format(sess.run(accuracy, feed_dict=val_feed)))
        
        if step > args.train_step:
            break
    time_end = time.time()
    print('=> training ends @ {}'.format(time_end))
    elapsed_time = time_end - time_start
    print('=> elapsed time for training: {}'.format(elapsed_time))

    # validate 
    val_feed = {x: mnist.validation.images, gt: mnist.validation.labels}
    print('=> validate accuracy: {}'.format(sess.run([accuracy], feed_dict=val_feed)))

if __name__ == '__main__':
    tf.app.run()
