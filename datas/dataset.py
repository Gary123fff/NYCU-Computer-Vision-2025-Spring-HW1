"""
This module contains a custom dataset class and utilities for loading image 
data, applying transformations, and handling class imbalances in training.
"""

from collections import Counter
from torchvision import datasets, transforms
import os
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader


class TestDataset(Dataset):
    """
    Custom dataset for loading images from a directory. This class allows 
    applying transformations and loading images with their respective filenames.
    """

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = sorted([os.path.join(data_dir, f) for f in os.listdir(
            data_dir) if f.endswith(('jpg', 'png'))])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, os.path.basename(img_path)


def get_transforms():
    """
    Returns a dictionary of data transformations for training and validation.

    The training transformations include random cropping, flipping, and 
    augmentation techniques. The validation transformations include resizing, 
    center cropping, and normalization.

    Returns:
        dict: A dictionary with 'train' and 'val' transformations.
    """
    return {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.AutoAugment(
                policy=transforms.AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])
    }


def get_balanced_dataloader(data_dir, transform, batch_size=64):
    """
    Returns a DataLoader for the dataset with class balancing using 
    WeightedRandomSampler.

    Args:
        data_dir (str): Path to the data directory.
        transform (callable): A function/transform to apply on the images.
        batch_size (int, optional): Batch size for loading the data (default is 64).

    Returns:
        dataloader: A PyTorch DataLoader instance.
        sorted_classes: Sorted list of class labels.
    """
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    sorted_classes = sorted(dataset.classes, key=int)
    class_to_idx = {cls_name: idx for idx,
                    cls_name in enumerate(sorted_classes)}
    dataset.class_to_idx = class_to_idx
    dataset.classes = sorted_classes

    targets = [sample[1] for sample in dataset.samples]
    class_count = Counter(targets)

    class_weights = {class_idx: 1.0/count for class_idx,
                     count in class_count.items()}
    weights = [class_weights[target] for target in targets]

    sampler = WeightedRandomSampler(weights, len(dataset), replacement=True)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4
    )

    return dataloader, sorted_classes


def prepare_test_data(test_path, data_transforms):
    """
    Prepares the test data for evaluation by loading the images from the given 
    path and applying the appropriate transformations.

    Args:
        test_path (str): Path to the test data directory.
        data_transforms (dict): Dictionary containing the transformation for 'val'.

    Returns:
        test_loader: A DataLoader for the test dataset.
    """
    test_transform = data_transforms["val"]
    test_dataset = TestDataset(test_path, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    return test_loader