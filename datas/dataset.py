import os
import torch
from collections import Counter
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class TestDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(('jpg', 'png'))])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, os.path.basename(img_path)

def get_transforms():
    return {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

def get_balanced_dataloader(data_dir, transform, batch_size=64):
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    sorted_classes = sorted(dataset.classes, key=lambda x: int(x))
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted_classes)}
    dataset.class_to_idx = class_to_idx
    dataset.classes = sorted_classes
    
    targets = [sample[1] for sample in dataset.samples]
    class_count = Counter(targets)
    
    class_weights = {class_idx: 1.0/count for class_idx, count in class_count.items()}
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
    test_transform = data_transforms["val"]
    test_dataset = TestDataset(test_path, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    return test_loader