import time
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class CifarModel():
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, device: torch.device = None):
        self.Model = model
        self.Optim = optimizer
        self.LossHist = []
        self.Epoch = 0
        self.device = device

    def predict(self, x: torch.Tensor):
        return self.Model(x)

    def score(self, data, label: np.ndarray):
        assert len(data) == len(label)
        preds = self.predict(torch.from_numpy(data).float())
        preds = [np.argmax(p.detach().numpy()) for p in preds]
        return np.sum(preds == label) / len(data)

    def train(self,
              maxEpochs, batchSize: int,
              criterion: nn.Module,
              data, label: torch.Tensor,
              writer: SummaryWriter = None):
        if writer is not None:
            for i, loss in enumerate(self.LossHist):
                writer.add_scalar('training loss', loss, i)

        for i in tqdm(range(maxEpochs)):

            startTime = time.time()
            perm = torch.randperm(len(data))

            numBatches = int(len(data)/batchSize + 1)
            totalLoss = 0
            for j in range(numBatches):
                self.Optim.zero_grad()

                indicies = perm[j*batchSize, j*batchSize+batchSize]

                preds = self.predict(data[indicies])
                loss = criterion.forward(preds, label[indicies])

                loss.backward()
                self.Optim.step()
                # print("  Loss {:.6f}".format(loss))
                totalLoss += loss

            elapsedTime = time.time() - startTime

            self.Epoch = self.Epoch+1

            avgLoss = totalLoss / numBatches
            print("Epoch {:d}/{:d}, Loss {:.6f}, Elapsed {:.3f} seconds...".format(i, maxEpochs, avgLoss, elapsedTime))
            if writer is not None:
                writer.add_scalar('training loss', avgLoss, self.Epoch)
            self.LossHist.append(avgLoss)

    def save(self, path: str):
        torch.save(
            {
                'epoch': self.Epoch,
                'model_state_dict': self.Model.state_dict(),
                'optimizer_state_dict': self.Optim.state_dict(),
                'loss': self.LossHist,
            },
            path
        )

    def load(self, path: str):
        checkpoint = torch.load(path)
        print(checkpoint)
        self.Model.load_state_dict(checkpoint['model_state_dict'])
        self.Optim.load_state_dict(checkpoint['optimizer_state_dict'])
        self.LossHist = checkpoint['loss']
        self.Epoch = checkpoint['epoch']
        self.Model.eval()