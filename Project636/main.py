import argparse
from routines import parseTrain
from test_routine import parseTest
from effnet_routine import parseEffNet

def parse():
    parser = argparse.ArgumentParser(description='Project636 Routines.')
    parser.add_argument('--device', dest='device', type=str, default='cpu',
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

    testParser = subparsers.add_parser('test', help='test help')
    parseTest(testParser)

    effNetParser = subparsers.add_parser('effnet', help='effnet help')
    parseEffNet(effNetParser)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse()
    args.func(args)
