import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime

if __name__ == '__main__':
    input = torch.zeros((1, 1, 32, 32))
    model = nn.Conv2d(1, 6, (3, 3), 1, 2)
    torch.onnx.export(model, input, "test_conv.onnx", verbose=True)
    model = onnx.load("test_conv.onnx")
    onnx.checker.check_model(model)
    onnx.helper.printable_graph(model.graph)