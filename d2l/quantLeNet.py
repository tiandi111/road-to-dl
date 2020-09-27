import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.quantization import QuantStub, DeQuantStub
from leNet import load_mnist

class QuantLeNet(nn.Module):
    def __init__(self):
        super(QuantLeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, (5, 5), 1, 2),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d((2, 2), 2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, (5, 5), 1, 0),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d((2, 2), 2)
        self.fc1 = nn.Sequential(
            nn.Linear(25 * 16, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = x.contiguous().view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def quant_forward(self, x):
        x = self.quant(x)
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = x.contiguous().view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.dequant(x)
        return x

    def fuse(self):
        for layer in [self.conv1, self.conv2]:
            torch.quantization.fuse_modules(layer, ['0', '1', '2'], inplace=True)
        for layer in [self.fc1, self.fc2]:
            torch.quantization.fuse_modules(layer, ['0', '1'], inplace=True)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def score(self, input, target):
        with torch.autograd.profiler.profile() as prof:
            st = time.time()
            preds = self.forward(input)
            correct = 0
            for i, pred in enumerate(preds):
                if pred.argmax() == target[i]:
                    correct += 1
            acc = correct / len(input)
            end = time.time()
            print("elapsed time:", end-st)
        print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        return acc

    def score_quant(self, input, target):
        with torch.autograd.profiler.profile() as prof:
            st = time.time()
            preds = self.quant_forward(input)
            correct = 0
            for i, pred in enumerate(preds):
                if pred.argmax() == target[i]:
                    correct += 1
            acc = correct / len(input)
            end = time.time()
            print("elapsed time:", end - st)
        print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        return acc

    def save_checkpoint(self, path:str, epoch, loss):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, path)

    def load_checkpoint(self, path:str):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.eval()
        return checkpoint

    def train1(self, optim, criterion, epochs, batch_size, train_data, train_label):
        for i in range(epochs):

            time_st = time.time()
            perm = torch.randperm(len(train_data))
            print("epoch:", i)

            for i in range(0, len(train_data), batch_size):
                optim.zero_grad()

                indicies = perm[i:i + batch_size]

                output = self.forward(train_data[indicies])
                loss = criterion.forward(output, train_label[indicies])

                loss.backward()
                optim.step()

            time_end = time.time()
            print("training time:", time_end - time_st)
            print("score:", self.score(train_data, train_label))




if __name__ == '__main__':
    train_data, train_label, test_data, test_label = load_mnist("/Users/tiandi03/Desktop/dataset")

    model = QuantLeNet()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # n_epochs = 10
    # batch_size = 128
    #
    # model.train1(optimizer, criterion, n_epochs, batch_size, train_data, train_label)
    #
    # model.save_checkpoint("pre_quant_model_%d.pkl"%(int(time.time())), n_epochs, 0)
    #
    model.load_checkpoint("/Users/tiandi03/road-to-dl/d2l/pre_quant_model_1599914926.pkl")

    # print(model)
    model.fuse()
    # print(model)

    model.qconfig = torch.quantization.default_qconfig
    # print(model.qconfig)
    torch.quantization.prepare(model, inplace=True)
    #
    print("score:", model.score(train_data, train_label))
    #
    torch.quantization.convert(model, inplace=True)
    # print(model)
    #
    print("quant score:", model.score_quant(train_data, train_label))
    model.save_checkpoint("post_quant_model_%d.pkl"%(int(time.time())), 10, 0)
