import argparse
import os

import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image
from dataset import CustomDataset

from models.vgg import *
from models.resnet import *


def main():

    print("=> creating vgg11 model ...")
    model = vgg11(num_classes=2)

    if torch.cuda.device_count() != 0:
        model = model.cuda()

    checkpoint = torch.load('./result/model.pth', map_location='cuda:0')
    cudnn.benchmark = True

    val_data = os.path.join('/home/bong08/tmp_lib/pytorch_practice/tools/tmp')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_loader = torch.utils.data.DataLoader(
        CustomDataset(val_data, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])))
    # batch_size = 32, shuffle = False,
    # num_workers = 4, pin_memory = True)

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    correct = 0
    all = 0
    for i,(images,target) in enumerate(val_loader):
        images, target = torch.autograd.Variable(images.float().cuda()), torch.autograd.Variable(target.long().cuda())
        output = model(images)
        _,predicted = torch.max(output,1)
        if target == predicted:
            correct += 1
        all += 1
    return correct/all


if __name__=='__main__':
    print(main())