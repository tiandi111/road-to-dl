import time
import torch
import numpy as np
import torch.jit as jit
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from preprocess import PreprocessImage, PreprocessImageBatch


class CifarModel():
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 optimizer: optim.Optimizer = None,
                 scheduler: optim.lr_scheduler = None):
        self.Model = model
        self.Optim = optimizer
        self.LossHist = []
        self.Epoch = 0
        self.device = device
        self.LRSchedule = []
        self.LRScheduler = scheduler

    def predict(self, x: torch.Tensor):
        return self.Model(x)

    def score(self, data, label: torch.Tensor, batchSize: int):
        assert len(data) == len(label)
        # stop auto-grad to save memory
        with torch.no_grad():
            label = label.cpu().detach().numpy()
            correct = 0
            for i in range(int(len(data) / batchSize) + 1):
                batchData = PreprocessImageBatch(data[i*batchSize: i*batchSize+batchSize], training=False)
                preds = self.predict(batchData.to(self.device).float())
                preds = [np.argmax(p) for p in preds.cpu().detach().numpy()]
                correct += np.sum(preds == label[i*batchSize: i*batchSize+batchSize])
            return correct / len(data)

    def train(self,
              maxEpochs, batchSize: int,
              criterion: nn.Module,
              trainData, trainLabel: torch.Tensor,
              validData, validLabel: torch.Tensor,
              writer: SummaryWriter = None):
        if writer is not None:
            for i, loss in enumerate(self.LossHist):
                writer.add_scalar('training loss', loss, i)
            for i, lr in enumerate(self.LRSchedule):
                writer.add_scalar('learning rate', lr, i)

        for i in tqdm(range(maxEpochs)):

            startTime = time.time()
            perm = torch.randperm(len(trainData))

            numBatches = int(len(trainData)/batchSize + 1)
            totalLoss = 0
            for j in range(numBatches):
                self.Optim.zero_grad()

                indicies = perm[j*batchSize: j*batchSize+batchSize]

                batchData = PreprocessImageBatch(trainData[indicies], training=True)

                preds = self.predict(batchData.to(self.device))
                loss = criterion.forward(preds, trainLabel[indicies])

                loss.backward()
                self.Optim.step()
                # print("  Loss {:.6f}".format(loss))
                totalLoss += float(loss)

            elapsedTime = time.time() - startTime

            if self.LRScheduler is not None:
                self.LRScheduler.step()

            self.Epoch = self.Epoch+1
            self.LRSchedule.append(self.Optim.param_groups[0]['lr'])

            avgLoss = totalLoss / numBatches
            print("Epoch {:d}/{:d}, Loss {:.6f}, Elapsed {:.3f} seconds...".format(i+1, maxEpochs, avgLoss, elapsedTime))
            # during validation, change to eval mode so that bn and dropout stop moving average
            self.Model.eval()
            print("Test score on validation set {:.3f}".format(self.score(validData, validLabel, batchSize)))
            self.Model.train()
            if writer is not None:
                writer.add_scalar('training loss', avgLoss, self.Epoch)
                writer.add_scalar('learning rate', self.Optim.defaults['lr'], self.Epoch)
            self.LossHist.append(avgLoss)

    def save(self, path: str):
        model = jit.script(self.Model)
        jit.save(model, path)

    def saveOnnx(self, path: str):
        input = torch.zeros(1, 3, 32, 32).to(self.device)
        output = self.Model(input)
        torch.onnx.export(self.Model, args=input, f=path, example_outputs=output, verbose=False)

    def saveCheckpoint(self, path: str):
        torch.save(
            {
                'epoch': self.Epoch,
                'model_state_dict': self.Model.state_dict(),
                # 'optimizer_state_dict': self.Optim.state_dict(),
                'loss': self.LossHist,
                'lr': self.LRSchedule,
            },
            path
        )

    def loadCheckpoint(self, path: str, training: bool):
        checkpoint = torch.load(path)
        self.Model.load_state_dict(checkpoint['model_state_dict'])
        # self.Optim.load_state_dict(checkpoint['optimizer_state_dict'])
        self.LossHist = checkpoint['loss']
        self.Epoch = checkpoint['epoch']
        self.LRSchedule = checkpoint['lr']
        if training:
            self.Model.train()
        else:
            self.Model.eval()
