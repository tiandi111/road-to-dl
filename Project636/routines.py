import torch
import time
import argparse
from network import ResNet
from model import CifarModel
from loader import loadData, trainValidSplit, localNorm
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


def parseTrain(parser: argparse.ArgumentParser):
    parser.add_argument('--lr', dest='lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--mom', dest='mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--wd', dest='wd', type=float, default=0.01, help='weight decay')

    parser.set_defaults(func=train)


def train(args):
    if args.device == 'gpu' and torch.cuda.is_available() is not True:
        print("cuda not available")
        exit(1)
    device = torch.device("cuda:0" if args.device == 'gpu' else "cpu")
    print("===\n"
          "Setup device: {d}".format(d=args.device))

    resNet = ResNet(stackSize=(2, 2, 2, 2)).to(device)
    # weight decay works as this in torch I guess: W(i+1) = （1 - weight_decay）* W（i）
    print("===\n"
          "Setup optimizer: \n"
          "learning rate: {lr}\n"
          "momentum: {mom}\n"
          "weight decay: {wd}".format(lr=args.lr, mom=args.mom, wd=args.wd))
    optimizer = optim.SGD(resNet.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.wd)
    tbWriter = SummaryWriter(args.tbDir)

    ResCifarModel = CifarModel(resNet, optimizer)
    if args.loadDir != '':
        print("===\n"
              "Loading model from {p}...".format(p=args.loadDir))
        ResCifarModel.load(args.loadDir)

    print("===\n"
          "Loading dataset from {p}...".format(p=args.dataDir))
    trainData, trainLabel, testData, testLabel = loadData(args.dataDir)
    trainData, trainLabel, validData, validLabel = trainValidSplit(trainData, trainLabel)
    trainData = localNorm(trainData)
    validData = localNorm(validData)

    criterion = nn.CrossEntropyLoss()
    print("===\n"
          "Start training...")
    ResCifarModel.train(maxEpochs=10, batchSize=128,
                        criterion=criterion,
                        data=torch.from_numpy(trainData).float().to(device),
                        label=torch.from_numpy(trainLabel).long().to(device),
                        writer=tbWriter)

    savePath = int(time.time())
    print("===\n"
          "Save model to {p}...".format(p=savePath))
    ResCifarModel.save("res_cifar_{:d}.pkl".format(savePath))
