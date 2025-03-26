"""
This script performs inference on test data using a trained ResNeXt model.
It loads the model, processes test data, generates predictions, and saves them to a CSV file.
"""
import torch
import pandas as pd
from tqdm import tqdm
import torchvision.datasets as datasets

from models.resxnet_model import create_resnext_model
from datas.dataset import prepare_test_data, get_transforms
from utils.metrics import plot_confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_model(model, test_loader, idx_to_class):
    """
    Runs the model on the test data and generates predictions.

    Args:
        model (torch.nn.Module): The trained model.
        test_loader (torch.utils.data.DataLoader): DataLoader for test data.
        idx_to_class (dict): Mapping of class indices to class labels.

    Returns:
        list: Predictions in the format [filename, predicted_label].
    """
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

def perform_inference(model_path, test_path, train_path):
    """
    Loads the model, prepares test data, and performs inference.

    Args:
        model_path (str): Path to the trained model.
        test_path (str): Path to the test data.
        train_path (str): Path to the training data (used for class-to-idx mapping).

    Returns:
        pd.DataFrame: DataFrame containing image names and predicted labels.
    """
    data_transforms = get_transforms()
    
    # Prepare test data
    test_loader = prepare_test_data(test_path, data_transforms)
    
    # Initialize model
    model = create_resnext_model(num_classes=100)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    
    # Get class mapping
    train_dataset = datasets.ImageFolder(train_path, transform=data_transforms['train'])
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
    
    # Predict
    predictions = test_model(model, test_loader, idx_to_class)

    # Save predictions
    df = pd.DataFrame(predictions, columns=['image_name', 'pred_label'])
    df.to_csv('prediction.csv', index=False)
    
    return df

def main():
    """
    Main function to perform inference using a trained model.
    """
    model_path = './improved_models/best_model8.pth'
    test_path = './data/test'
    train_path = './data/train'
    
    perform_inference(model_path, test_path, train_path)

if __name__ == '__main__':
    """
    Run the main function to perform inference.
    """
    main()