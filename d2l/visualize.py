import leNet
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

if __name__ == '__main__':
    net = leNet.LeNet()
    net.load_checkpoint("/Users/tiandi03/road-to-dl/d2l/model_lasso_1599761993.pkl")
    # for i, f in enumerate(net.conv2.weight):
    #     print("filter: %d"%i, f.norm())
    # print(net.conv2.weight)
    # # net.load_checkpoint("/Users/tiandi03/road-to-dl/d2l/model_1599756910.pkl")
    # # print(net.conv1.weight.norm())
    # net.visualize_kernel()
    train_data, train_label, test_data, test_label = leNet.load_mnist("/Users/tiandi03/Desktop/dataset")
    net.visualize_output(train_data[0:1])
