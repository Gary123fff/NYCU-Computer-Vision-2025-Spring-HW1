"""
This script handles both training and inference for the model. 
It can execute either the training or inference process based on the argument passed.
"""

from train import main as train_main
from inference import main as inference_main

def main(train=False):
    """
    Executes the training or inference process depending on the value of the 'train' argument.
    
    Args:
        train (bool): If True, runs the training process. If False, runs the inference process.
    """
    if train:
        train_main()
    else:
        inference_main()

if __name__ == '__main__':
    main(train=False)
