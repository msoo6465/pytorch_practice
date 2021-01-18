# codes from torchvision.datasets.ImageFolder

import os

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import imgaug.augmenters as iaa

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

"""
커스텀 데이터 셋은 총 3가지의 기본 오버라이딩 함수로 이루어져 있다.
__init__ , __getitem__, __len__
__init__ : 데이터의 위치와 타겟의 라벨링을 하는 부분
__getitem__ : 데이터의 위치에 해당하는 실제 이미지 값을 불러와 이미지 변환을 시켜주는 부분.
"""

class ImageFolder(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, loader=pil_loader, extensions=IMG_EXTENSIONS, is_valid_file=None):
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.extensions = extensions
        self.is_valid_file = is_valid_file

        # read directory
        label_list = os.listdir(self.root)

        try:
            # convert to number(integer)(ex: 0, 1, 2, 3 ...)
            label_list = [int(l) for l in label_list]
        except:
            # if there is a non-integer directory(skip conversion)
            pass

        # sort label list
        label_list = sorted(label_list)

        # read label directories
        samples = []
        for index, label in enumerate(label_list):
            label_dir = os.path.join(self.root, str(label))

            # read label directory
            for filename in os.listdir(label_dir):
                if not os.path.splitext(filename)[-1].lower() in self.extensions:
                    # skip invalid extensions
                    continue

                # append to sample (image_path, label)
                ## sample의 형태 (image_path, index) index는 라벨에 맞는 번호로써 학습이나 밸리데이션 하는 부분에서 비교 하기 위해 label을 index화 시킴
                samples.append((os.path.join(label_dir, filename), index))  # Caution: we use index as label, no directory name

        if not self.is_valid_file:
            self.imgaug = iaa.Sequential([
                iaa.Fliplr(0.5), # horizontal flips
                iaa.Crop(percent=(0, 0.1)), # random crops
                # Small gaussian blur with random sigma between 0 and 0.5.
                # But we only blur about 50% of all images.
                iaa.Sometimes(
                    0.5,
                    iaa.GaussianBlur(sigma=(0, 0.5))
                ),
                # Strengthen or weaken the contrast in each image.
                iaa.LinearContrast((0.75, 1.5)),
                # Add gaussian noise.
                # For 50% of all images, we sample the noise once per pixel.
                # For the other 50% of all images, we sample the noise per pixel AND
                # channel. This can change the color (not only brightness) of the
                # pixels.
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                # Make some images brighter and some darker.
                # In 20% of all cases, we sample the multiplier once per channel,
                # which can end up changing the color of the images.
                iaa.Multiply((0.8, 1.2), per_channel=0.2),
                # Apply affine transformations to each image.
                # Scale/zoom them, translate/move them, rotate them and shear them.
                iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    rotate=(-25, 25),
                    shear=(-8, 8)
                )
            ], random_order=True)
        self.samples = samples
        self.targets = [s[1] for s in samples]

        print('Labels(ordered):', label_list)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]

        # read image
        sample = self.loader(path)

        # apply transform to input(image)
        if self.transform is not None:
            sample = self.transform(sample)

        # apply transform to target(label)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # os.makedirs('tmp',exist_ok=True)
        # mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=torch.float32).reshape(3, 1, 1)
        # std = torch.as_tensor([0.229, 0.224, 0.225], dtype=torch.float32).reshape(3, 1, 1)
        # sample_cpu = sample.mul_(std).add_(mean).mul_(255).clamp_(0, 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        # Image.fromarray(sample_cpu).save(os.path.join('tmp', os.path.basename(path)))

        return sample, target

    def __len__(self):
        return len(self.samples)
