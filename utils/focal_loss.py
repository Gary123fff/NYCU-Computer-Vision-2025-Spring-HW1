import torch
from torch import nn
import torch.nn.functional as F


"""
This module implements the Focal Loss function and a combined criterion 
for training deep learning models.
"""


class FocalLoss(nn.Module):
    """
    Focal Loss function implementation for addressing class imbalance.

    Args:
        alpha (float): Weighting factor for the class.
        gamma (float): Focusing parameter that reduces the loss for well-classified examples.
        reduction (str): Specifies the reduction to apply to the output: 
            'none' | 'mean' | 'sum'. Default is 'mean'.
    """

    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Compute the Focal Loss between the inputs and targets.

        Args:
            inputs (Tensor): The model's output logits.
            targets (Tensor): The ground truth labels.

        Returns:
            Tensor: The calculated Focal Loss.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        if self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def create_combined_criterion(outputs, targets):
    """
    Compute a combined loss function that is a weighted sum of Focal Loss and 
    Cross-Entropy Loss with label smoothing.

    Args:
        outputs (Tensor): The model's output logits.
        targets (Tensor): The ground truth labels.

    Returns:
        Tensor: The calculated combined loss.
    """
    return (
        FocalLoss(gamma=2)(outputs, targets) * 0.8 +
        nn.CrossEntropyLoss(label_smoothing=0.1)(outputs, targets) * 0.2
    )
