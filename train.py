"""
This script trains and evaluates a model using ResNeXt and several augmentation techniques.
It supports early stopping, learning rate scheduling, and model saving.
"""

import os
import torch
import random
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from models.resxnet_model import create_resnext_model
from datas.dataset import get_balanced_dataloader, get_transforms
from utils.focal_loss import create_combined_criterion
from utils.metrics import EarlyStopping
from utils.augmentations import mixup_data
from utils.augmentations import cutmix_data
from utils.metrics import plot_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, train_loader, val_loader, criterion, epochs=100, save_path='./models'):
    """
    Train a model using ResNeXt with various augmentation techniques.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): Training dataset loader.
        val_loader (DataLoader): Validation dataset loader.
        criterion (function): Loss function.
        epochs (int, optional): Number of training epochs. Default is 100.
        save_path (str, optional): Directory to save models. Default is './models'.

    Returns:
        tuple: Lists of train/val losses and accuracies.
    """
    os.makedirs(save_path, exist_ok=True)
    best_acc = 0.0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    model = model.to(device)
    optimizer = AdamW([
        {'params': [p for n, p in model.named_parameters() if 'layer2' in n],
         'lr': 2e-4},
        {'params': [p for n, p in model.named_parameters() if 'layer3' in n],
         'lr': 3e-4},
        {'params': [p for n, p in model.named_parameters() if 'layer4' in n],
         'lr': 5e-4},
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

    early_stopping = EarlyStopping(
        patience=15, verbose=True, path=os.path.join(save_path, 'early_stop_model.pth'))

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        model.train()
        running_loss, correct, total = 0.0, 0.0, 0

        progress_bar = tqdm(
            train_loader, desc=f"Training Epoch {epoch+1}", leave=False)

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            augmentation_prob = 0.3
            if random.random() < augmentation_prob:
                if random.random() < 0.5:
                    inputs, targets_a, targets_b, lam = mixup_data(
                        inputs, labels, alpha=0.5)
                else:
                    inputs, targets_a, targets_b, lam = cutmix_data(
                        inputs, labels, alpha=0.5)

                outputs = model(inputs)
                loss = lam * criterion(outputs, targets_a) + \
                    (1 - lam) * criterion(outputs, targets_b)

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
            progress_bar.set_postfix(loss=loss.item(), acc=(
                correct.double() / total).item())

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
            torch.save(model.state_dict(), os.path.join(
                save_path, 'best_model8.pth'))
            print('Best model saved!')

        if epoch % 10 == 0 or epoch == epochs - 1:
            plot_metrics(train_losses, val_losses,
                         train_accuracies, val_accuracies, save_path)

    return train_losses, val_losses, train_accuracies, val_accuracies


def main():
    train_path = "./data/train"
    val_path = "./data/val"
    data_transforms = get_transforms()
    train_loader, _ = get_balanced_dataloader(
        train_path, data_transforms['train'])
    val_loader, _ = get_balanced_dataloader(val_path, data_transforms['val'])

    model = create_resnext_model(num_classes=100)
    criterion = create_combined_criterion

    train_model(model, train_loader, val_loader, criterion,
                epochs=1000, save_path='./improved_models')


if __name__ == '__main__':
    main()
