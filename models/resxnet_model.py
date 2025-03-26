from torch import nn
from torchvision.models import resnext101_64x4d, ResNeXt101_64X4D_Weights


"""
This module contains the definition of a modified ResNeXt-101 model with 
customized layers for transfer learning.
"""


class GlobalAvgPool(nn.Module):
    """
    This class implements global average pooling followed by flattening of the 
    input tensor. It reduces the spatial dimensions to a single value per 
    channel and flattens the result.
    """

    def forward(self, x):
        """
        Forward pass for global average pooling. 
        Reduces the spatial dimensions to 1x1 per channel and flattens.

        Args:
            x (Tensor): The input tensor to apply the pooling operation.

        Returns:
            Tensor: The pooled and flattened tensor.
        """
        return nn.functional.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)


def create_resnext_model(num_classes=100):
    """
    Creates a ResNeXt-101 model pre-trained on ImageNet with specific 
    customizations such as freezing early layers and modifying the 
    fully connected layer for a new number of output classes.

    Args:
        num_classes (int): The number of classes for the final classification 
                            layer (default is 100).

    Returns:
        nn.Module: The modified ResNeXt-101 model.
    """
    weights = ResNeXt101_64X4D_Weights.IMAGENET1K_V1
    base_model = resnext101_64x4d(weights=weights)

    # Freeze most layers
    for param in base_model.parameters():
        param.requires_grad = False

    # Unfreeze specific layers
    for name, param in base_model.named_parameters():
        if "layer2" in name or "layer3" in name or "layer4" in name:
            param.requires_grad = True

    for param in base_model.fc.parameters():
        param.requires_grad = True

    base_model.avgpool = GlobalAvgPool()
    in_features = base_model.fc.in_features
    base_model.fc = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(1024, num_classes)
    )

    return base_model
