import torch
import argparse
import torch.jit as jit
from model import CifarModel
from loader import loadData
import time


def parseTest(parser: argparse.ArgumentParser):
    parser.add_argument('--target', dest='target',
                        type=str, choices=['train', 'test'],
                        default='train', help='target testing data set')

    parser.set_defaults(func=test)


def test(args):
    if args.device == 'gpu' and torch.cuda.is_available() is not True:
        print("cuda not available")
        exit(1)
    device = torch.device("cuda:0" if args.device == 'gpu' else "cpu")
    print("===\n"
          "Setup device: {d}".format(d=args.device))

    model = jit.load(args.loadDir)
    model.eval()
    ResCifarModel = CifarModel(model, device=device, optimizer=None)

    print("===\n"
          "Loading dataset from {p}...".format(p=args.dataDir))
    trainData, trainLabel, testData, testLabel = loadData(args.dataDir)

    score = 0
    st = time.time()
    if args.target == 'train':
        score = ResCifarModel.score(data=torch.from_numpy(trainData).float(),
                            label=torch.from_numpy(trainLabel).long(),
                            batchSize=128)
    else:
        score = ResCifarModel.score(data=torch.from_numpy(testData).float(),
                            label=torch.from_numpy(testLabel).long(),
                            batchSize=128)
    print("===\n"
          "Score: {:.3f}\n"
          "Time: {:.3f}".format(score, time.time()-st))
