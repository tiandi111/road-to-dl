import torch
import time
import argparse
from network import ResNet
from model import CifarModel
from loader import loadData, trainValidSplit
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

    return parser.parse_args()

if __name__ == '__main__':
    args = parse()

    if args.device == 'gpu' and torch.cuda.is_available() is not True:
        print("cuda not available")
        exit(1)
    device = torch.device("cuda:0" if args.device == 'gpu' else "cpu")

    ResNet = ResNet(stackSize=(2, 2, 2, 2)).to(device)
    optimizer = optim.SGD(ResNet.parameters(), lr=0.001, momentum=0.9, weight_decay=0.9)
    tbWriter = SummaryWriter(args.tbDir)

    ResCifarModel = CifarModel(ResNet, optimizer)
    if args.loadDir != '':
        ResCifarModel.load(args.loadDir)

    trainData, trainLabel, testData, testLabel = loadData(args.dataDir)
    trainData, trainLabel, validData, validLabel = trainValidSplit(trainData, trainLabel)

    criterion = nn.CrossEntropyLoss()
    ResCifarModel.train(maxEpochs=10, batchSize=128,
                        criterion=criterion,
                        data=torch.from_numpy(trainData).float().to(device), label=torch.from_numpy(trainLabel).long().to(device),
                        writer=tbWriter)

    ResCifarModel.save("res_cifar_{:d}.pkl".format(int(time.time())))
