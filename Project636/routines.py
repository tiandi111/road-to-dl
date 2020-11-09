import torch
import time
import argparse
import torch.jit as jit
from network import ResNet
from model import CifarModel
from loader import loadData, trainValidSplit, localNorm
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


def parseTrain(parser: argparse.ArgumentParser):
    parser.add_argument('--ep', dest='ep', type=int, default=20, help='max epochs')
    parser.add_argument('--lr', dest='lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--mom', dest='mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--wd', dest='wd', type=float, default=0, help='weight decay')
    parser.add_argument('--batch', dest='batch', type=int, default=128, help='batch size')
    parser.add_argument('--validSplit', dest='validSplit', type=int, default=10, help='train-valid split percent')
    parser.add_argument('--firstNumFilters', dest='firstNumFilters', type=int, default=16,
                        help='number of filters at the first layer')
    parser.add_argument('--arch', dest='arch', type=str, default='2,2,2,2',
                        help='network architecture')

    parser.set_defaults(func=train)


def train(args):
    if args.device == 'gpu' and torch.cuda.is_available() is not True:
        print("cuda not available")
        exit(1)
    device = torch.device("cuda:0" if args.device == 'gpu' else "cpu")
    print("===\n"
          "Setup device: {d}".format(d=args.device))

    arch = [int(l) for l in args.arch.split(',')]
    resNet = ResNet(firstNumFilter=args.firstNumFilters, stackSize=arch).to(device)
    # weight decay works as this in torch I guess: W(i+1) = （1 - weight_decay）* W（i）
    print("===\n"
          "Setup trainer: \n"
          "network architecture: {arc}\n"
          "max epochs: {ep}\n"
          "batch size: {bs}\n"
          "initial learning rate: {lr}\n"
          "momentum: {mom}\n"
          "weight decay: {wd}\n"
          "number of filters at the first layer: {numf}".
          format(arc=arch, ep=args.ep, bs=args.batch, lr=args.lr, mom=args.mom, wd=args.wd, numf=args.firstNumFilters))
    optimizer = optim.SGD(resNet.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[60, 120, 140, 160, 175, 190], gamma=0.1)
    tbWriter = SummaryWriter(args.tbDir)

    ResCifarModel = CifarModel(resNet, optimizer=optimizer, device=device, scheduler=scheduler)
    if args.loadDir != '':
        print("===\n"
              "Loading model from {p}...".format(p=args.loadDir))
        ResCifarModel.loadCheckpoint(args.loadDir, training=True)

    print("===\n"
          "Loading dataset from {p}...".format(p=args.dataDir))
    trainData, trainLabel, testData, testLabel = loadData(args.dataDir)
    splitIndex = int(len(trainData)*(1-args.validSplit*0.01))
    print("{:d} images will be reserved as validation set".format(len(trainData)-splitIndex))
    trainData, trainLabel, validData, validLabel = trainValidSplit(trainData, trainLabel, split_index=splitIndex)
    trainData = localNorm(trainData)
    validData = localNorm(validData)

    criterion = nn.CrossEntropyLoss()
    print("===\n"
          "Start training...")
    ResCifarModel.train(maxEpochs=args.ep, batchSize=args.batch,
                        criterion=criterion,
                        trainData=torch.from_numpy(trainData).float(),#.to(device),
                        trainLabel=torch.from_numpy(trainLabel).long().to(device),
                        validData=torch.from_numpy(validData).float(),#.to(device),
                        validLabel=torch.from_numpy(validLabel).long().to(device),
                        writer=tbWriter)

    timestamp = int(time.time())
    print("===\n"
          "Save checkpoint to res_cifar_{p}.pkl...".format(p=timestamp))
    ResCifarModel.saveCheckpoint("res_cifar_{:d}.pkl".format(timestamp))
    print("===\n"
          "Save model to res_cifar_{p}.jit...".format(p=timestamp))
    ResCifarModel.save("res_cifar_{:d}.jit".format(timestamp))
    print("===\n"
          "Save onnx model to res_cifar_{p}.onnx...".format(p=timestamp))
    ResCifarModel.saveOnnx("res_cifar_{:d}.onnx".format(timestamp))




