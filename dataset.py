import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os




from PIL import Image

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class CustomDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None, loader=pil_loader, is_valid_file=False):
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        files = os.listdir(self.root)
        samples = []
        if is_valid_file == False:
            for filename in files:
                print(filename)
                if filename.split('.')[0]=='cat':
                    label = 0
                else:
                    label = 1
                samples.append((os.path.join(self.root, filename), label))
        self.samples = samples
        self.targets = [s[1] for s in samples]


    def __getitem__(self, index):
        path, target = self.samples[index]

        # read image
        sample = self.loader(path)

        # apply transform to input(image)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

# if __name__=='__main__':
#     # model = resnet34(num_classes=1)
#     # criterion = nn.CrossEntropyLoss()
#     # optimizer = torch.optim.SGD(model.parameters(), 0.01, momentum=0.9, weight_decay=1e-4)
#     #
#     train_data = os.path.join('/home/nexys/PycharmProjects/pytorch_practice/data/train')
#     # val_data = os.path.join('/home/nexys/PycharmProjects/cnn_practice/data','val')
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#
#     train_dataset = CustomDataset(train_data,transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         normalize,
#     ]))
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True,num_workers=4, pin_memory=True)
    # val_loader = torch.utils.data.DataLoader(
    #     CustomDataset(val_data, transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=32, shuffle=False,
    #     num_workers=4, pin_memory=True)
    # for epoch in range(50):
    #     print(f'epoch: {epoch}')
    #     for i,(images,target) in enumerate(train_loader):
    #         output = model(images)
    #         loss = criterion(output,target)
    #
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    # correct = 0
    # incorrect = 0
    # for i,(images,target) in enumerate(val_loader):
    #     model.eval()
    #     output = model(images)
    #
    #     if output==target:
    #         correct +=1
    #     else:
    #         incorrect +=1
    # print((correct-incorrect)/(correct+incorrect))