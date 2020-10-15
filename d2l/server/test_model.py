import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime

if __name__ == '__main__':
    input = torch.zeros((1, 1, 4, 4))
    model = nn.Sequential(
        nn.Conv2d(1, 2, (3, 3), 1, 1),
        nn.BatchNorm2d(2),
    )
    torch.onnx.export(model, input, "test_conv.onnx", verbose=True)
    model = onnx.load("test_conv.onnx")
    onnx.checker.check_model(model)
    onnx.helper.printable_graph(model.graph)