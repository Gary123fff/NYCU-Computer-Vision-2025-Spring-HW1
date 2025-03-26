# NYCU-Computer-Vision-2025-Spring-HW1
313551139 陳冠豪
## Introduction
# Image Classification Model using ResNeXt-101_64x4d

This repository contains an image classification model built using the pretrained `ResNeXt-101_64x4d` model. The goal is to achieve high performance on image classification tasks by applying advanced techniques such as data augmentation, class balancing, and smart training strategies.

 *Model Architecture*

- **Backbone**: `ResNeXt-101_64x4d` from `torchvision.models` is used as the backbone, providing strong feature extraction capabilities.
- **Global Average Pooling**: The model ends with Global Average Pooling (GlobalAvgPool) to handle variable input image sizes effectively.

 *Loss Function*

- **Focal Loss**: This loss function is used to address class imbalance by focusing more on hard-to-classify samples, improving model performance on imbalanced datasets.

 *Data Augmentation*

- **MixUp & CutMix**: These techniques are applied to blend images and labels, helping the model generalize better.
- **AutoAugment**: This method introduces more variations to the training images, further improving the model's robustness.

 *Data Loading*

- **Balanced Dataloader**: The `get_balanced_dataloader()` function uses `WeightedRandomSampler` to ensure class balance in each batch, improving performance on imbalanced datasets.

 *Training Process*

- **Optimizer**: AdamW optimizer is used, with different learning rates for different layers in ResNeXt-101 to optimize performance.
- **OneCycleLR**: This learning rate scheduler dynamically adjusts the learning rate during training to help the model avoid local minima.
- **Early Stopping**: The model monitors validation loss and stops training early if overfitting is detected, ensuring better generalization.

 *Goal*

The model is designed to improve generalization by utilizing smart data augmentation strategies, dynamic learning rate adjustment, and handling class imbalance through Focal Loss. These techniques combine to create a robust image classification model that performs well on challenging datasets.
