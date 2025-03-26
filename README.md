# NYCU-Computer-Vision-2025-Spring-HW1
313551139 陳冠豪
## Introduction
For this Homework, we’re building an image classification model .
To achieve good results, I used the pretrained ResNeXt-101_64x4d model and applied the following techniques to improve performance:
Model Architecture:
•	Uses torchvision.models.resnext101_64x4d as the backbone.
•	Ends with Global Average Pooling (GlobalAvgPool) to handle different input image sizes.
Loss Function:
•	Uses Focal Loss, which helps the model focus more on hard-to-classify samples, making it great for handling imbalanced datasets.
Data Augmentation:
•	Applies MixUp and CutMix to make the model generalize better by blending images and labels.
•	Uses AutoAugment to introduce more variations in the training images.
Data Loading:
•	The function get_balanced_dataloader() includes WeightedRandomSampler to ensure class balance in each batch.
Training Process:
•	Uses AdamW optimizer, with different learning rates for different layers in ResNeXt-101.
•	Uses OneCycleLR to dynamically adjust the learning rate, helping the model avoid getting stuck in local minima.
•	Implements Early Stopping to monitor validation loss and stop training early if needed to prevent overfitting.
The goal is to improve the model’s generalization by using smart data augmentation and learning rate strategies, while Focal Loss helps handle class imbalance.
