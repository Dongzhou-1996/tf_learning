#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser("TF cluster")
# create TF clusters' parameters
parser.add_argument('--task_index', default=None, type=int, 
                                    help='worker task index should >= 0.' \
                                        'task_index=0 is the master worker task that performs the initialization of variables.')
parser.add_argument('--ps_hosts', default=None, type=str, help='comma-separated list of hostname:port pairs')
parser.add_argument('--worker_hosts', default=None, type=str, help='comma-separated list of hostname:port pairs')
parser.add_argument('--job_name', type=str, default=None, choices=['worker', 'PS'])


def main():
    PS_spec = args.ps_hosts.split(',')
    worker_spec = args.worker_hosts.split(',')
    # define TF cluster
    cluster= tf.train.ClusterSpec({'PS': PS_spec, 'work': worker_spec})
    server = tf.train.Server(cluster, job_name=args.job_name, task_index=args.task_index)
    if args.job_name=='PS':
        print('=> start server thread ...')
        server_join()
    is_chief = (args.task_index == 0)


