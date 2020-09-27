import time
import torch
import torch.nn as nn
import torch.nn.quantized as qt
from torch.quantization import QuantStub, DeQuantStub, QuantWrapper

class Bencher(nn.Module):
    def __init__(self):
        super(Bencher, self).__init__()
        self.conv = nn.Conv2d(1, 6, 3, 1, 1)
        self.qunat_conv = qt.Conv2d(1, 6, 3, 1, 1)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.qconfig = torch.quantization.default_qconfig

    def forward(self, input):
        x = self.quant(input)
        x = self.qunat_conv(x)
        return self.dequant(x)

    def forward_cmp(self, input):
        return self.conv(input)

if __name__ == '__main__':

    b = Bencher()
    max_iter = 5000
    input = torch.rand(16, 1, 28, 28)

    st = time.time()
    for i in range(max_iter):
        b.forward_cmp(input)
    end = time.time()
    print("normal conv:", end - st)

    torch.quantization.prepare(b, inplace=True)
    torch.quantization.convert(b, inplace=True)

    st = time.time()
    for i in range(max_iter):
        b.forward(input)
    end = time.time()
    print("quant conv:", end-st)