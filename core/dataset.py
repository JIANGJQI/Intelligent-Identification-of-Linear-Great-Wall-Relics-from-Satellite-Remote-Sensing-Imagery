import torchvision.transforms.functional as F
import torch
import numpy as np
import random
import os
import glob
from PIL import Image
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from torchvision import transforms


class ToTensor(object):

    def __call__(self, data):
        image, label = data['image'], data['label']
        return {'image': F.to_tensor(image), 'label': F.to_tensor(label)}


class Resize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        image, label = data['image'], data['label']

        return {'image': F.resize(image, self.size), 'label': F.resize(label, self.size, interpolation=InterpolationMode.BICUBIC)}


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']

        if random.random() < self.p:
            return {'image': F.hflip(image), 'label': F.hflip(label)}

        return {'image': image, 'label': label}


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']

        if random.random() < self.p:
            return {'image': F.vflip(image), 'label': F.vflip(label)}

        return {'image': image, 'label': label}


class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = F.normalize(image, self.mean, self.std)
        return {'image': image, 'label': label}
    

class FullDataset(Dataset):
    def __init__(self, image_roots, gt_roots, size, mode):
        self.images = []
        self.gts = []
        for img_root, gt_root in zip(image_roots, gt_roots):
            imgs = sorted([img_root + f for f in os.listdir(img_root) if f.endswith('.jpg') or f.endswith('.png')])
            gts_  = sorted([gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')])
            self.images.extend(imgs)
            self.gts.extend(gts_)
        if mode == 'train':
            self.transform = transforms.Compose([
                Resize((size, size)),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                ToTensor(),
                Normalize()
            ])
        else:
            self.transform = transforms.Compose([
                Resize((size, size)),
                ToTensor(),
                Normalize()
            ])

    def __getitem__(self, idx):
        image = self.rgb_loader(self.images[idx])
        label = self.binary_loader(self.gts[idx])
        data = {'image': image, 'label': label}
        data = self.transform(data)
        return data

    def __len__(self):
        return len(self.images)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
        

class SAM2TestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        print(f"找到 {len(self.image_paths)} 个测试图像")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, img_path