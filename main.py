import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from torch.utils.data import WeightedRandomSampler, DataLoader, Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import AdamW
from torchvision.models import resnext101_64x4d, ResNeXt101_64X4D_Weights
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from PIL import Image
from tqdm import tqdm
from collections import Counter
import matplotlib
matplotlib.use('Agg')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class GlobalAvgPool(nn.Module):
    def forward(self, x):
        return nn.functional.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)

class EarlyStopping:
    def __init__(self, patience=15, verbose=True, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased. Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

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

def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size, _, H, W = x.shape
    index = torch.randperm(batch_size).to(x.device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(H, W, lam)
    
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    y_a, y_b = y, y[index]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))
    return x, y_a, y_b, lam

def rand_bbox(H, W, lam):
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

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

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, save_path):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Train Accuracy')
    plt.plot(epochs, val_accuracies, 'r-', label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'best_model_v8.png'))
    plt.close()

def train_model(model, train_loader, val_loader, criterion, epochs=100, save_path='./models'):
    os.makedirs(save_path, exist_ok=True)
    best_acc = 0.0
    
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    
    
    model = model.to(device)
    
    optimizer = AdamW([
        {'params': [p for n, p in model.named_parameters() if 'layer2' in n], 'lr': 2e-4},
        {'params': [p for n, p in model.named_parameters() if 'layer3' in n], 'lr': 3e-4},
        {'params': [p for n, p in model.named_parameters() if 'layer4' in n], 'lr': 5e-4},
        {'params': model.fc.parameters(), 'lr': 1e-3}
    ], weight_decay=5e-4)
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[2e-4, 3e-4, 5e-4, 1e-3],
        steps_per_epoch=len(train_loader),
        epochs=1000,
        pct_start=0.2,
        div_factor=25,
        final_div_factor=1e4,
        anneal_strategy='cos'
    )
    
    early_stopping = EarlyStopping(patience=15, verbose=True, path=os.path.join(save_path, 'early_stop_model.pth'))
    
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        model.train()
        running_loss, correct, total = 0.0, 0.0, 0
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False)
        
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            augmentation_prob = 0.3
            if random.random() < augmentation_prob:
                if random.random() < 0.5:
                    inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=0.5)
                else:
                    inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels, alpha=0.5)
                
                outputs = model(inputs)
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)

                _, preds = torch.max(outputs, 1)
                correct_a = torch.sum(preds == targets_a)
                correct_b = torch.sum(preds == targets_b)
                batch_correct = lam * correct_a + (1 - lam) * correct_b
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                batch_correct = torch.sum(preds == labels.data)
                
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            correct += batch_correct
            total += labels.size(0)
            
            progress_bar.set_postfix(loss=loss.item(), acc=(correct.double() / total).item())
        
        epoch_loss = running_loss / total
        epoch_acc = correct.double() / total
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc.item())
        
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels.data)
                total += labels.size(0)
        
        val_loss = val_loss / total
        val_acc = correct.double() / total
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        
        val_losses.append(val_loss)
        val_accuracies.append(val_acc.item())
        
        scheduler.step()
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping triggered. Training stopped.")
            break
      
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model8.pth'))
            print('Best model saved!')
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, save_path)
    
    return train_losses, val_losses, train_accuracies, val_accuracies

def test_model(model, test_loader, idx_to_class):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for inputs, filenames in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for filename, pred in zip(filenames, preds.cpu().numpy()):
                filename = filename.replace('.jpg', '')
                class_label = idx_to_class[pred]
                predictions.append([filename, class_label])

    return predictions

def prepare_test_data(test_path, data_transforms):
    test_transform = data_transforms["val"]
    test_dataset = TestDataset(test_path, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    return test_loader

def perform_inference(model, test_loader, train_path, data_transforms):
    train_dataset = datasets.ImageFolder(train_path, transform=data_transforms['train'])
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
    
    model.load_state_dict(torch.load('./improved_models/best_model8.pth'))
    predictions = test_model(model, test_loader, idx_to_class)
    
    df = pd.DataFrame(predictions, columns=['image_name', 'pred_label'])
    df.to_csv('prediction.csv', index=False)
    
    return df

def main(train):
    train_path = "./data/train"
    val_path = "./data/val"
    test_path = "./data/test"
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.3),
            AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
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
    
    train_loader, class_names = get_balanced_dataloader(train_path, data_transforms['train'])
    val_loader, _ = get_balanced_dataloader(val_path, data_transforms['val'])
    
    weights = ResNeXt101_64X4D_Weights.IMAGENET1K_V1
    base_model = resnext101_64x4d(weights=weights)
    
    for param in base_model.parameters():
        param.requires_grad = False
    
    for name, param in base_model.named_parameters():
        if "layer2" in name or "layer3" in name or "layer4" in name:
            param.requires_grad = True
    
    for param in base_model.fc.parameters():
        param.requires_grad = True
    
    num_classes = 100
    base_model.avgpool = GlobalAvgPool()
    in_features = base_model.fc.in_features
    base_model.fc = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(1024, num_classes)
    )
    
    criterion = lambda outputs, targets: (
        FocalLoss(gamma=2)(outputs, targets) * 0.8 + 
        nn.CrossEntropyLoss(label_smoothing=0.1)(outputs, targets) * 0.2
    )
    if train :
        train_model(
            base_model, 
            train_loader, 
            val_loader, 
            criterion, 
            epochs=1000,
            save_path='./improved_models/best_model8.pth'
        )
    
   
    test_loader = prepare_test_data(test_path, data_transforms)
    predictions_df = perform_inference(base_model.to(device), test_loader, train_path, data_transforms)

if __name__ == '__main__':
    main(train=False)