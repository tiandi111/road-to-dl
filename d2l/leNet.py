import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx as onnx
import matplotlib.pyplot as plt

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, (5, 5), 1, 2),
            nn.BatchNorm2d(6),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d((2, 2), 2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, (5, 5), 1, 0),
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

    def forward(self, x) -> torch.Tensor:
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = x.contiguous().view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def score(self, input, target):
        with torch.autograd.profiler.profile() as prof:
            preds = self.forward(input)
        print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        correct = 0
        for i, pred in enumerate(preds):
            if pred.argmax() == target[i]:
                correct += 1
        return correct/len(input)

    def filter_wise_l1(self):
        group_l1 = 0
        for layer in [self.conv1[0], self.conv2[0]]:
            for filter in layer.weight:
                group_l1 += filter.norm()
        return group_l1

    def channel_wise_l1(self):
        group_l1 = 0
        for layer in [self.conv1[0], self.conv2[0]]:
            for c in range(layer.weight.shape[1]):
                group_l1_c = 0
                for filter in layer.weight:
                    group_l1_c += filter[c].norm()**2
                group_l1 += math.sqrt(group_l1_c)
        return group_l1

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

    def export_as_onnx(self):
        onnx.export(self,
                    torch.randn(1, 1, 28, 28),
                    "lenet.onnx",
        )
        return

    def visualize_kernel(self):
        for i, layer in enumerate([self.conv1[0], self.conv2[0]]):
            n_filters, n_channels = layer.weight.shape[0:2]
            idx = 0
            fig, axs = plt.subplots(n_filters, n_channels)
            axs = axs.flat
            print("layer:%d"%i)
            for j, filter in enumerate(layer.weight):
                print("filter:%d norm:"%j, filter.norm())
                for k, channel in enumerate(filter):
                    axs[idx].set_yticklabels([])
                    axs[idx].set_xticklabels([])
                    axs[idx].imshow(channel.detach().numpy(), cmap="gray")
                    idx += 1
                    print("channel:%d norm:"%k, channel.norm())
                print("\n")
            fig.show()
            # filter-wise norm hist
            f_norms = np.asarray([f.norm().detach().numpy() for f in layer.weight])
            plt.figure()
            plt.hist2d(np.arange(len(f_norms)), f_norms, bins=2*len(f_norms))
            # channel-wise norm hist
            c_norms = np.zeros(n_channels)
            for i in range(n_channels):
                for j in range(n_filters):
                    c_norms[i] += layer.weight[j][i].norm()**2
                c_norms[i] = math.sqrt(c_norms[i])
            plt.figure()
            plt.hist2d(np.arange(len(c_norms)), c_norms, bins=2*len(c_norms))
        plt.show()

    def visualize_output(self, input):
        for layer in [self.conv1[0], self.conv2[0]]:
            n_filters = layer.weight.shape[0]
            output = layer(input)
            fig, axs = plt.subplots(n_filters, 1)
            idx = 0
            axs = axs.flat
            for f_out in output[0]:
                axs[idx].set_yticklabels([])
                axs[idx].set_xticklabels([])
                axs[idx].imshow(f_out.detach().numpy(), cmap="gray")
                idx += 1
                print(f_out.norm())
            fig.show()
            input = output
        plt.show()

def one_hot_mnist_label(label_arr:bytes):
    label = torch.zeros((len(label_arr), 10), dtype=torch.float)
    for i, l in enumerate(label_arr):
        label[i][l] = 1
    return label

def load_mnist(data_path:str):
    train_data_str = open(data_path + "/train-images-idx3-ubyte", mode="rb").read()
    train_data = torch.tensor(list(train_data_str[16:]), dtype=torch.float).reshape((60000, 1, 28, 28))

    train_label_str = open(data_path + "/train-labels-idx1-ubyte", mode="rb").read()
    # train_label = one_hot_mnist_label(train_label_str[8:])
    train_label = torch.tensor(list(train_label_str[8:]), dtype=torch.long)

    test_data_str = open(data_path + "/t10k-images-idx3-ubyte", mode="rb").read()
    test_data = torch.tensor(list(test_data_str[16:]), dtype=torch.float).reshape((10000, 1, 28, 28))

    test_label_str = open(data_path + "/t10k-labels-idx1-ubyte", mode="rb").read()
    # test_label = one_hot_mnist_label(test_label_str[8:])
    test_label = torch.tensor(list(test_label_str[8:]), dtype=torch.long)

    print("load dataset successfully")
    return train_data, train_label, test_data, test_label


if __name__ == '__main__':
    train_data, train_label, test_data, test_label = load_mnist("/Users/tiandi03/Desktop/dataset")

    net = LeNet()
    checkpoint = net.load_checkpoint("/Users/tiandi03/road-to-dl/d2l/model_l1_1599978601.pkl")
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    n_epochs = 1
    batch_size = 128

    lambda_f = 0.1
    lambda_c = 0.1

    for i in range(n_epochs):

        time_st = time.time()

        perm = torch.randperm(len(train_data))
        print("epoch:", i)

        for i in range(0, len(train_data), batch_size):
            optimizer.zero_grad()

            indicies = perm[i:i+batch_size]

            output = net.forward(train_data[indicies])
            loss = criterion.forward(output, train_label[indicies]) + lambda_f*net.filter_wise_l1() + lambda_c*net.channel_wise_l1()

            loss.backward()
            optimizer.step()

        time_end = time.time()
        print("training time:", time_end - time_st)
        print("score:", net.score(train_data, train_label))

    # net.save_checkpoint("model_lasso_%d.pkl"%(int(time.time())), checkpoint["epoch"]+n_epochs, loss)
    net.save_checkpoint("model_l1_%d.pkl"%(int(time.time())), n_epochs, loss)