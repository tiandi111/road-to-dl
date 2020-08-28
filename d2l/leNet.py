import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, (5, 5), 1, 2)
        self.pool1 = nn.MaxPool2d((2, 2), 2)
        self.conv2 = nn.Conv2d(6, 16, (5, 5), 1, 0)
        self.pool2 = nn.MaxPool2d((2, 2), 2)
        self.fc1 = nn.Linear(25 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


    def forward(self, x) -> torch.Tensor:
        x = self.pool1(F.sigmoid(self.conv1(x)))
        x = self.pool2(F.sigmoid(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def score(self, input, target):
        preds = self.forward(input)
        correct = 0
        for i, pred in enumerate(preds):
            if pred.argmax() == target[i].argmax():
                correct += 1
        return correct/len(input)

def one_hot_mnist_label(label_arr:bytes):
    label = torch.zeros((len(label_arr), 10), dtype=torch.float)
    for i, l in enumerate(label_arr):
        label[i][l] = 1
    return label

def load_mnist(data_path:str):
    train_data_str = open(data_path + "/train-images-idx3-ubyte", mode="rb").read()
    train_data = torch.tensor(list(train_data_str[16:]), dtype=torch.float).reshape((60000, 1, 28, 28))

    train_label_str = open(data_path + "/train-labels-idx1-ubyte", mode="rb").read()
    train_label = one_hot_mnist_label(train_label_str[8:])

    test_data_str = open(data_path + "/t10k-images-idx3-ubyte", mode="rb").read()
    test_data = torch.tensor(list(test_data_str[16:]), dtype=torch.float).reshape((10000, 1, 28, 28))

    test_label_str = open(data_path + "/t10k-labels-idx1-ubyte", mode="rb").read()
    test_label = one_hot_mnist_label(test_label_str[8:])

    print("load dataset successfully")
    return train_data, train_label, test_data, test_label

if __name__ == '__main__':
    train_data, train_label, test_data, test_label = load_mnist("/Users/tiandi03/Desktop/dataset")

    net = LeNet()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.MSELoss()

    n_epochs = 100
    batch_size = 128

    for i in range(n_epochs):

        perm = torch.randperm(len(train_data))
        print("epoch:", i)

        for i in range(0, len(train_data), batch_size):
            optimizer.zero_grad()

            indicies = perm[i:i+batch_size]

            output = net.forward(train_data[indicies])
            loss = criterion.forward(output, train_label[indicies])

            loss.backward()
            optimizer.step()

        print("score:", net.score(test_data, test_label))