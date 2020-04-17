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
