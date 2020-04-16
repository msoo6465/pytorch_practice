import torch
from dataset import CustomDataset
from models.resnet import *
from torch import nn
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import shutil

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth')


def main():
    model = resnet34(num_classes=1)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 0.01, momentum=0.9, weight_decay=1e-4)

    train_data = os.path.join('/home/nexys/PycharmProjects/pytorch_practice/data/train')
    val_data = os.path.join('/home/nexys/PycharmProjects/cnn_practice/data','val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = CustomDataset(train_data,transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True,num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CustomDataset(val_data, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=32, shuffle=False,
        num_workers=4, pin_memory=True)
    for epoch in range(50):
        print(f'epoch: {epoch}')
        for i,(images,target) in enumerate(train_loader):
            output = model(images)
            loss = criterion(output,target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    os.makedirs('./result',exist_ok=True)
    torch.save(model.state_dict(), './result')