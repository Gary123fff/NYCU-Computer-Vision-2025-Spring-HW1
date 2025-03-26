import torch
import numpy as np


"""
This module implements data augmentation techniques such as MixUp and CutMix 
for training deep learning models.
"""


def mixup_data(x, y, alpha=0.4):
    """
    Apply the MixUp augmentation technique to input data.

    Args:
        x (Tensor): Input data (e.g., images).
        y (Tensor): Corresponding labels.
        alpha (float): The parameter that controls the distribution of mixing (default=0.4).

    Returns:
        Tensor: Mixed input data.
        Tensor: The original label for the first sample.
        Tensor: The original label for the second sample.
        float: The mixing coefficient lambda.
    """
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
    """
    Apply the CutMix augmentation technique to input data.

    Args:
        x (Tensor): Input data (e.g., images).
        y (Tensor): Corresponding labels.
        alpha (float): The parameter that controls the distribution of mixing (default=0.4).

    Returns:
        Tensor: Augmented input data.
        Tensor: The original label for the first sample.
        Tensor: The original label for the second sample.
        float: The mixing coefficient lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size, _, height, width = x.shape
    index = torch.randperm(batch_size).to(x.device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(height, width, lam)

    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    y_a, y_b = y, y[index]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (height * width))
    return x, y_a, y_b, lam


def rand_bbox(height, width, lam):
    """
    Generate random bounding box coordinates for the CutMix operation.

    Args:
        height (int): Height of the input image.
        width (int): Width of the input image.
        lam (float): The mixing coefficient lambda.

    Returns:
        tuple: Bounding box coordinates (bbx1, bby1, bbx2, bby2).
    """
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(width * cut_rat)
    cut_h = int(height * cut_rat)
    cx = np.random.randint(width)
    cy = np.random.randint(height)
    bbx1 = np.clip(cx - cut_w // 2, 0, width)
    bby1 = np.clip(cy - cut_h // 2, 0, height)
    bbx2 = np.clip(cx + cut_w // 2, 0, width)
    bby2 = np.clip(cy + cut_h // 2, 0, height)
    return bbx1, bby1, bbx2, bby2
