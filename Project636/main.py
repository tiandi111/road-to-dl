import torch
import time
import argparse
from routines import parseTrain, train
from network import ResNet
from model import CifarModel
from loader import loadData, trainValidSplit, localNorm
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

def parse():
    parser = argparse.ArgumentParser(description='Project636 Routines.')
    parser.add_argument('--device', dest='device', type=str, choices=['cpu', 'gpu'], default='cpu',
                        help='choose device to run the routines')
    parser.add_argument('--tensorboard_dir', dest='tbDir', type=str, default='./log',
                        help='tensorboard directory')
    parser.add_argument('--load', dest='loadDir', type=str, default='',
                        help='model directory')
    parser.add_argument('--data_dir', dest='dataDir', type=str, default='/Users/tiandi03/Desktop/dataset/cifar-10-batches-py',
                        help='model directory')

    subparsers = parser.add_subparsers(help="sub-command help")

    trainParser = subparsers.add_parser('train', help='train help')
    parseTrain(trainParser)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse()
    args.func(args)