import torch
import torch.nn as nn
import torchvision

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, (3, 3), 1, 1, bias=True),
            nn.BatchNorm2d(3, momentum=0)
        )
        self.fc = nn.Linear(48, 5, bias=True)
        self.fc.weight.data.fill_(0)
        self.fc.bias.data.fill_(0)

    def forward(self, x):
        y = self.conv(x)
        # y = y.view(1, 48)
        # y = self.fc(y)
        return y

if __name__ == '__main__':
    model = TestModel()

    dummy_input = torch.zeros(1, 1, 4, 4)

    print(model.forward(dummy_input))

    torch.onnx.export(model, dummy_input, "/Users/tiandi03/road-to-dl/d2l/server/test/models/export.onnx", verbose=True)