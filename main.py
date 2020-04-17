import torch
from dataset import CustomDataset
from models.resnet import *
from torch import nn
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import shutil
from models.vgg import *

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth')


def main():
    model = vgg11(num_classes=2)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), 0.01, momentum=0.9, weight_decay=1e-4)

    train_data = os.path.join('/home/bong08/tmp_lib/pytorch_practice/data/train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = CustomDataset(train_data,transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True,num_workers=4, pin_memory=True)

    model.train()
    for epoch in range(50):
        print(f'epoch : {epoch}')
        for i,(images,target) in enumerate(train_loader):
            images,target = torch.autograd.Variable(images.float().cuda()),torch.autograd.Variable(target.long().cuda())
            output = model(images)
            loss = criterion(output,target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    os.makedirs('./result',exist_ok=True)
    torch.save({'state_dict':model.module.state_dict()}, './result/model.pth')

if __name__=='__main__':
    main()