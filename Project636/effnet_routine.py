import torch
import argparse
import subprocess


def parseEffNet(parser: argparse.ArgumentParser):
    parser.add_argument('--df', dest='depths', default='', type=str, help='depth scaling factor')
    parser.add_argument('--wf', dest='widths', default='', type=str, help='width scaling factor')
    parser.add_argument('--baseDepth', dest='baseDepth', default=16, type=int, help='baseline depth')
    parser.add_argument('--baseWidth', dest='baseWidth', default=32, type=int, help='baseline width')

    parser.set_defaults(func=EffNetGridSearch)


def EffNetGridSearch(args):
    if args.device == 'gpu' and torch.cuda.is_available() is not True:
        print("cuda not available")
        exit(1)
    device = torch.device("cuda:0" if args.device == 'gpu' else "cpu")
    print("===\n"
          "Setup device: {d}".format(d=args.device))

    depths = [float(d) for d in args.depths.split(',')]
    widths = [float(w) for w in args.widths.split(',')]
    print("===\n"
          "Scaling factor options: \n"
          "depth: ", depths,
          "\nwidth: ", widths)

    for df in depths:
        for wf in widths:
            layers = 16-int(16*df)
            arch = [3, 4, 6, 3]
            for i in range(layers):
                arch[i % 4] -= 1
            arch = [str(i) for i in arch]
            cmd = ("python ./main.py --data_dir={data} --device={dev} train "
                      "--firstNumFilters={fil} "
                      "--ep=150 "
                      "--lr=0.1 "
                      "--wd=1e-4 "
                      "--arch={arc}".format(
                data=args.dataDir, dev=args.device, fil=int(wf*args.baseWidth), arc=','.join(arch)
            ))
            subprocess.Popen(cmd, shell=True)


