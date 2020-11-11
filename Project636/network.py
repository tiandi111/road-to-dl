import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """
    inC == outC
    if downSampling is True, will insert a (1, 1) Conv2d layer at the head of the block
    """
    def __init__(self, inC, bottleC, outC: int, downSampling=False):
        super(ResBlock, self).__init__()
        self.Block = nn.Sequential(
            nn.BatchNorm2d(inC),
            nn.ReLU(),
            nn.Conv2d(inC, bottleC, (1, 1), 2 if downSampling else 1, 0),

            nn.BatchNorm2d(bottleC),
            nn.ReLU(),
            nn.Conv2d(bottleC, bottleC, (3, 3), 1, 1),

            nn.BatchNorm2d(bottleC),
            nn.ReLU(),
            nn.Conv2d(bottleC, outC, (1, 1), 1, 0),
        )
        if inC != outC:
            self.Shortcut = nn.Conv2d(inC, outC, (1, 1), 2 if downSampling else 1, 0)
        else:
            self.Shortcut = None

    def forward(self, X: torch.Tensor):
        if self.Shortcut is not None:
            return self.Block(X) + self.Shortcut(X)
        return self.Block(X) + X

class ResNet(nn.Module):
    """
        Network Architecture:
            (3, 3) conv, 64, /2

            (64, 64, 64)
    """
    def __init__(self, firstNumFilter: int, stackSize = (2, 2, 2, 2)):
        super(ResNet, self).__init__()
        self.InConv = nn.Conv2d(3, firstNumFilter, (3, 3), 1, 1)
        self.ResPart = nn.Sequential()
        self.lastOutC = firstNumFilter
        for i in range(len(stackSize)):
            outC = 4 * firstNumFilter * (2**i)
            bottC = int(outC / 4)
            for j in range(stackSize[0]):
                self.ResPart.add_module("stack{i}_{j}".format(i=i, j=j),
                                        ResBlock(self.lastOutC, bottC, outC, downSampling=(i > 0 and j == 0)))
                self.lastOutC = outC
        self.FC = nn.Linear(self.lastOutC, 10)

    """
    Args: 
        x: torch.Tensor of shape [batchSize, Channel, Height, Weight]
    """
    def forward(self, x: torch.Tensor):
        x = self.InConv(x)
        x = self.ResPart(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.FC(x.view(int(x.size()[0]), int(x.size()[1])))
        return x


