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
    parser.add_argument('--ep', dest='ep', type=int, default=20, help='max epochs')
    parser.add_argument('--lr', dest='lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--mom', dest='mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--wd', dest='wd', type=float, default=0, help='weight decay')

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
          "Setup trainer: \n"
          "max epochs: {ep}\n"
          "learning rate: {lr}\n"
          "momentum: {mom}\n"
          "weight decay: {wd}".format(ep=args.ep, lr=args.lr, mom=args.mom, wd=args.wd))
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
    ResCifarModel.train(maxEpochs=args.ep, batchSize=128,
                        criterion=criterion,
                        trainData=torch.from_numpy(trainData[:2]).float().to(device),
                        trainLabel=torch.from_numpy(trainLabel[:2]).long().to(device),
                        validData=torch.from_numpy(validData[:2]).float().to(device),
                        validLabel=torch.from_numpy(validLabel[:2]).long().to(device),
                        writer=tbWriter)

    savePath = int(time.time())
    print("===\n"
          "Save model to {p}...".format(p=savePath))
    ResCifarModel.save("res_cifar_{:d}.pkl".format(savePath))

# def parseTest(parser: argparse.ArgumentParser):
#     parser.set_defaults(func=train)

# def test():