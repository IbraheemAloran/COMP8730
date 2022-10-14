import numpy as np
import torch
import torchvision.datasets as ds
import torchvision.transforms as tf
from torch.autograd import Variable
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
train = ds.MNIST(root='mnistDataset/', train=True, transform=tf.ToTensor(), download=True)
test = ds.MNIST(root='mnistDataset/', train=False, transform=tf.ToTensor(), download=True)


loadTrain = DataLoader(train, batch_size=batch, shuffle=True)
loadTest = DataLoader(test, batch_size=batch, shuffle=True)

#print(loadTrain)
#print(test.data.size())

#plt.imshow(loadTrain.data[0], cmap='gray')
#plt.show()

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.convLay1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))

        self.convLay2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))

        self.out = nn.Linear(32*7*7, 10)

    def forward(self, x):
        x = self.convLay1(x)
        x = self.convLay2(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)

        return x


cnn = CNN()
print(cnn)

lossFn = nn.CrossEntropyLoss()
optim = op.Adam(cnn.parameters(), lr = learningRate)


def trainModel():
    #cnn.train()
    for epoch in range(epochs):
        for i, (img, label) in enumerate(loadTrain):
            img = img.to(device=device)
            label = label.to(device=device)
            output = cnn(img)
            loss = lossFn(output, label)
            optim.zero_grad()
            loss.backward()
            optim.step()
            print("Training: ", i, "/60000. Loss: ", loss)

def testModel():
    for img, label in loadTest:
        result = cnn(img)
        print(result)



trainModel()


