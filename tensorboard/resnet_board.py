import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import torch.nn as nn
import shutil
import tqdm
import random
import torch.backends.cudnn as cudnn


writer = SummaryWriter()

epochs = 30
best_acc1 = 0
lr = 0.1

transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
trainset = datasets.MNIST('mnist_train',train=True,download=False,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size =32,shuffle=True)
valset = datasets.MNIST('mnist_val',train=False,download=True,transform=transform)
valloader = torch.utils.data.DataLoader(valset,batch_size=32,shuffle=False)
model = torchvision.models.resnet50(False,num_classes=10)

random.seed(1)
torch.manual_seed(1)
cudnn.deterministic = True

model.conv1 = torch.nn.Conv2d(1,64, kernel_size=7,stride=2,padding=3,bias=False)
images, labels = next(iter(trainloader))

criterion = nn.CrossEntropyLoss().cuda()
model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr ,momentum=0.9,weight_decay=1e-4)


## make_grid는 배치를 나누어 하나의 이미지로 저장
gpu = 0

resume = "checkpoint.pth.tar"
grid = torchvision.utils.make_grid(images)

loc = 'cuda:{}'.format(gpu)
checkpoint = torch.load(resume, map_location=loc)

## map_location은 어떤 gpu에 로드 할 것인지
start_epoch = checkpoint['epoch']
best_acc1 = checkpoint['best_acc1']
# best_acc1 may be from a checkpoint from a different GPU
best_acc1 = best_acc1.to(gpu)
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))

def adjust_learning_rate(optimizer, epoch):
    global lr
    lr = lr*(0.1 **(epoch//10))
    for param_group in optimizer.param_groups:
        param_group['lr']= lr


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def validation(val_loader, model, criterion, topk = (1,)):
    model.eval()

    with torch.no_grad():
        for i,(img,tar) in enumerate(val_loader):
            img = img.cuda(non_blocking=True)
            tar = tar.cuda(non_blocking=True)
            output = model(img)
            with torch.no_grad():
                maxk = max(topk)
                batch_size = tar.size(0)
                loss = criterion(output,tar)
                _, pred = output.topk(maxk, 1, True, True)
                ## t()는 전치
                print(pred)
                pred = pred.t()
                correct = pred.eq(tar.view(1, -1).expand_as(pred))

                res = []
                for k in topk:
                    correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                    res.append(correct_k.mul_(100.0 / batch_size))
                print(res)
                return res[0]

# writer.add_image('images',grid,0)
# writer.add_graph(model,images)
writer.close()

for i in range(epochs):
    for ind,(img, tar) in enumerate(trainloader):

        img = img.cuda( non_blocking=True)
        tar = tar.cuda( non_blocking=True)
        output = model(img)
        loss = criterion(output,tar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ind % 20 == 0:
            print(f"{ind}/{len(trainloader)}")


    acc1 = validation(valloader, model, criterion, topk=(1,1))
    is_best = acc1 > best_acc1
    best_acc1 = max(acc1, best_acc1)

    save_checkpoint({
        'epoch': i + 1,
        'arch': 'resnet50',
        'state_dict': model.state_dict(),
        'best_acc1': best_acc1,
        'optimizer': optimizer.state_dict(),
    }, is_best)




