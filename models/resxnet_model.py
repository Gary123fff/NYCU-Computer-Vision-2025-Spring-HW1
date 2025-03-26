import torch
import torch.nn as nn
from torchvision.models import resnext101_64x4d, ResNeXt101_64X4D_Weights

class GlobalAvgPool(nn.Module):
    def forward(self, x):
        return nn.functional.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)

def create_resnext_model(num_classes=100):
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