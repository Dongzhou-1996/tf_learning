#!/usr/bin/env python
# coding=utf-8
import argparse
parser = argparse.ArgumentParser('demo')
parser.add_argument('--model_path', type=str, default='/home', required=True, 
                    help='the path of checkpoint file')
parser.add_argument('--model_name', type=str, default='yolov3',
                    choices=['yolov3', 'yolov5', 'fcos'], dest='model')
parser.add_argument('--version', action='version', version='{} v1.0'.format(parser.prog))
parser.print_help()
args, unparsed = parser.parse_known_args()

print(args)
print(unparsed)


