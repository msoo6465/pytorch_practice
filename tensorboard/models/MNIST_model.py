import torch
import torch.nn as nn

class MNIST_model(nn.Module):
    def __init__(self):
        super(MNIST_model, self).__init__()
        self.keep_prob = 0.5
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            nn.ReLU()
        )
        self.maxPool2d = nn.MaxPool2d(kernel_size=2,stride=2,padding=1)

        self.fc1 = nn.Linear(4*4*128,625,bias=True)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.layer4 = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(p= 1 - self.keep_prob)
        )
        self.fc2 = nn.Linear(625,10,bias=True)
        nn.init.xavier_uniform_(self.fc2.weight)
        ## 기존의 무작위 수로 초기화하는 것과는 다르게 layer의 특성에 맞게 초기화 하는 방법

    def activations_hook(self,grad):
        self.gradients = grad

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        h = out.register_hook(self.activations_hook)
        out = self.maxPool2d(out)
        out = out.view(out.size(0),-1)
        out = self.layer4(out)
        out = self.fc2(out)
        return out

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self,x):
        return self.layer3(self.layer2(self.layer1(x)))