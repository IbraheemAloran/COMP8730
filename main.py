import numpy as np
import torch
import torchvision.datasets as ds
import torchvision.transforms as tf
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as op

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparameters
batch = 64
epochs = 1
learningRate = 0.001
classes = 10


#load the datasets
train = ds.MNIST(root='mnistDataset/', train=True, transform=tf.ToTensor())
test = ds.MNIST(root='mnistDataset/', train=False, transform=tf.ToTensor())

#
#loadTrain = DataLoader(dataset='mnistDataset/', batch_size=batch, shuffle=True)
#loadTest = DataLoader(dataset='mnistDataset/', batch_size=batch, shuffle=True)

print(train.data.size())
print(test.data.size())

plt.imshow(train.data[0], cmap='gray')
plt.show()

class CNN(nn.Module):
    def __int__(self):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Linear(64 * 7 * 7, 10)
        )

    def forwardFeed(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        out = self.out(x)

        return out, x


cnn = CNN()
print(cnn)



loss = nn.CrossEntropyLoss()
optim = op.Adam(cnn.parameters(), lr = learningRate)

