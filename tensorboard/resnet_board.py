import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import torch.nn as nn
import shutil
import tqdm

writer = SummaryWriter()

epochs = 30
best_acc1 = 0

transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
trainset = datasets.MNIST('mnist_train',train=True,download=False,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size =32,shuffle=True)
valset = datasets.MNIST('mnist_val',train=False,download=True,transform=transform)
valloader = torch.utils.data.DataLoader(valset,batch_size=32,shuffle=False)
model = torchvision.models.resnet50(False,num_classes=10)

model.conv1 = torch.nn.Conv2d(1,64, kernel_size=7,stride=2,padding=3,bias=False)
images, labels = next(iter(trainloader))

criterion = nn.CrossEntropyLoss().cuda()
model.cuda()
optimizer = torch.optim.SGD(model.parameters(), 0.05,
                                momentum=0.9,
                                weight_decay=1e-4)
## make_grid는 배치를 나누어 하나의 이미지로 저장
grid = torchvision.utils.make_grid(images)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def validation(val_loader, model, criterion):
    model.eval()

    with torch.no_grad():
        for img,tar in val_loader:
            img = img.cuda(non_blocking=True)
            tar = tar.cuda(non_blocking=True)
            output = model(img)

            loss = criterion(output,tar)
            _, pred = output.topk(1,1,True,True)
            pred = pred.t()
            correct = pred.eq(tar.view(1, -1).expand_as(pred))
            correct_k = correct.reshape(-1).float().sum(0, keepdim=True)
            print(f"Loss : {loss}")
    return correct_k

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


    acc1 = validation(valloader, model, criterion)
    is_best = acc1 > best_acc1
    best_acc1 = max(acc1, best_acc1)

    save_checkpoint({
        'epoch': i + 1,
        'arch': 'resnet50',
        'state_dict': model.state_dict(),
        'best_acc1': best_acc1,
        'optimizer': optimizer.state_dict(),
    }, is_best)




