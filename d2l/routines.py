import leNet
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import onnx
import onnxruntime

def onnx_routine(model_path:str, data_path:str):
    m = onnx.load(model_path)
    onnx.checker.check_model(m)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    inf_sess = onnxruntime.InferenceSession(model_path)
    train_data, train_label, test_data, test_label = leNet.load_mnist(data_path)

    print(to_numpy(train_data[0]).shape)
    inp = {inf_sess.get_inputs()[0].name: to_numpy(train_data[0:1])}
    out = inf_sess.run(None, inp)
    print(out)
    print(train_label[0])

if __name__ == '__main__':
    onnx_routine("lenet.onnx", "/Users/tiandi03/Desktop/dataset")

# if __name__ == '__main__':
#     net = leNet.LeNet()
#     net.load_checkpoint("/Users/tiandi03/road-to-dl/d2l/model_l1_1599979969.pkl")
#
#     # net.visualize_kernel()
#     train_data, train_label, test_data, test_label = leNet.load_mnist("/Users/tiandi03/Desktop/dataset")
    # print("score:", net.score(test_data, test_label))
#     # net.visualize_output(train_data[0:1])
