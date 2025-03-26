import os
import torch
import numpy as np
import matplotlib.pyplot as plt

""" This module contains utility functions and classes for early stopping 
and plotting training/validation metrics. """

class EarlyStopping:
    """
    EarlyStopping monitors the validation loss and stops training if it 
    does not improve after a specified number of epochs (patience). 
    Attributes:
        patience (int): Number of epochs to wait for improvement before stopping.
        verbose (bool): Whether to print messages during early stopping.
        delta (float): Minimum change in validation loss to qualify as an improvement.
        path (str): Path to save the best model.
        counter (int): Counter to track the number of epochs without improvement.
        best_score (float): Best validation score achieved so far.
        early_stop (bool): Flag indicating if early stopping should occur.
        val_loss_min (float): Minimum validation loss observed.
    """

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
        """
        Check if the early stopping condition is met and save the model checkpoint.
 
        Args:
            val_loss (float): The current validation loss.
            model (nn.Module): The PyTorch model to save if validation loss decreases.
        """
        score = -val_loss 
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print('EarlyStopping counter:',
                      self.counter, 'out of', self.patience)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Save the model checkpoint if validation loss has decreased.
 
        Args:
            val_loss (float): The current validation loss.
            model (nn.Module): The PyTorch model to save.
        """
        if self.verbose:
            print('Validation loss decreased. Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, save_path):
    """
    Plot and save training and validation losses and accuracies.
 
    Args:
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses.
        train_accuracies (list): List of training accuracies.
        val_accuracies (list): List of validation accuracies.
        save_path (str): Path to save the plot image.
    """
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
