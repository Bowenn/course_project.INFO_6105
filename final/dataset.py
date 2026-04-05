import os
import glob

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


def get_train_transforms(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class CrackDataset(Dataset):
    """Load images from Positive/ and Negative/ subdirectories."""

    LABEL_MAP = {"Negative": 0, "Positive": 1}

    def __init__(self, file_list, label_list, transform=None):
        self.files = file_list
        self.labels = label_list
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


class TestDataset(Dataset):
    """Load test images from a flat directory."""

    def __init__(self, test_dir, transform=None):
        self.files = sorted(
            glob.glob(os.path.join(test_dir, "*.*")),
            key=lambda x: os.path.basename(x),
        )
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        image_id = os.path.splitext(os.path.basename(path))[0]
        return img, image_id


def build_file_list(data_dir):
    """Return (file_paths, labels) by scanning Positive/ and Negative/ subdirs."""
    files, labels = [], []
    for class_name, label in CrackDataset.LABEL_MAP.items():
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Expected directory: {class_dir}")
        for f in glob.glob(os.path.join(class_dir, "*.*")):
            files.append(f)
            labels.append(label)
    return files, labels
