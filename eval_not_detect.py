import tensorflow as tf
import os
import json
import subprocess
from scipy.misc import imread, imresize
from scipy import misc
from shutil import copyfile

from train import build_forward
from utils.annolist import AnnotationLib as al
from utils.train_utils import add_rectangles, rescale_boxes

import cv2
import argparse

def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--true_boxes', required=True)
    parser.add_argument('--weights', default='./output/results')
    parser.add_argument('--expname', default='')
    #parser.add_argument('--test_boxes', required=True)
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--logdir', default='output')
    parser.add_argument('--iou_threshold', default=0.5, type=float)
    parser.add_argument('--tau', default=0.25, type=float)
    parser.add_argument('--min_conf', default=0.2, type=float)
    parser.add_argument('--show_suppressed', default=True, type=bool)
    parser.add_argument("--add_title", default="Models-trained-on-Brainwash-Mrsub-dataset-Tested-on-Clifton-test-dataset", type=str)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    expname = args.expname + '_' if args.expname else ''
    if not os.path.exists(args.weights):
        os.makedirs(args.weights)


    try:
        #rpc_cmd = './utils/annolist/doRPC.py --minOverlap %f %s %s' % (args.iou_threshold, true_boxes, pred_boxes)
        #print('$ %s' % rpc_cmd)
        #rpc_output = subprocess.check_output(rpc_cmd, shell=True)
        #print(rpc_output)
        #txt_file = [line for line in rpc_output.split('\n') if line.strip()][-1]
        txt_file_1='/home/vivalab/lstm_latest/output/results/LSTM-mobilenet.txt'
        txt_file_3='/home/vivalab/lstm_latest/output/results/LSTM-incepV1.txt'
        txt_file_5='/home/vivalab/lstm_latest/output/results/LSTM-resnet-50.txt'
        txt_file_7='/home/vivalab/lstm_latest/output/results/LSTM-resnet-101.txt'
        txt_file_4='/home/vivalab/lstm_latest/output/results/LSTM-incep-resnetV2.txt'
        txt_file_6='/home/vivalab/lstm_latest/output/results/LSTM-incepV3.txt'
        txt_file_2='/home/vivalab/lstm_latest/output/results/LSTM-squeezenet.txt'
        output_png = '%s/results.png' % args.weights
        plot_cmd = './utils/annolist/plotSimple.py %s %s %s %s %s %s %s --output-file %s --title %s' % (txt_file_2, txt_file_3, txt_file_1, txt_file_5, txt_file_6, txt_file_7, txt_file_4, output_png, args.add_title )
        print('$ %s' % plot_cmd)
        plot_output = subprocess.check_output(plot_cmd, shell=True)
        print('output results at: %s' % plot_output)
    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()