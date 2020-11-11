import torch
import argparse
import subprocess


def parseEffNet(parser: argparse.ArgumentParser):
    parser.add_argument('--root', dest='rootDir', default='.', type=str, help='project root directory')
    parser.add_argument('--df', dest='depths', default='', type=str, help='depth scaling factor')
    parser.add_argument('--wf', dest='widths', default='', type=str, help='width scaling factor')
    parser.add_argument('--baseDepth', dest='baseDepth', default=16, type=int, help='baseline depth')
    parser.add_argument('--baseWidth', dest='baseWidth', default=32, type=int, help='baseline width')

    parser.set_defaults(func=EffNetGridSearch)


def EffNetGridSearch(args):
    if args.device == 'gpu' and torch.cuda.is_available() is not True:
        print("cuda not available")
        exit(1)
    device = args.device
    print("===\n"
          "Setup device: {d}".format(d=args.device))

    depths = [float(d) for d in args.depths.split(',')]
    widths = [float(w) for w in args.widths.split(',')]
    assert len(depths) == len(widths)
    print("===\n"
          "Scaling factor options: \n"
          "depth: ", depths,
          "\nwidth: ", widths)

    for i in range(len(depths)):
        thisDev = torch.device(i%torch.cuda.device_count() if device=='gpu' else 'cpu')
        layers = 33-int(33 * depths[i] + 0.5)
        arch = [3, 4, 23, 3]
        if layers >= 33:
            print("invalid depth factor, not enough layers to remove")
            return
        if layers <= 19:
            arch[2] -= layers
        else:
            arch[2] -= 19
            for k in range(layers-19):
                arch[k%4] -= 1
        arch = [str(i) for i in arch]
        cmd = ("python {rt}/main.py --data_dir={data} --device={dev} train "
                  "--firstNumFilters={fil} "
                  "--ep=150 "
                  "--lr=0.09 "
                  "--wd=1e-4 "
                  "--arch={arc}".format(
            rt=args.rootDir, data=args.dataDir, dev=thisDev, fil=int(widths[i] * args.baseWidth + 0.5), arc=','.join(arch)
        ))
        subprocess.Popen(cmd, shell=True)


