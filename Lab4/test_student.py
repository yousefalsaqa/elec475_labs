import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
from model import student  # Import your custom model
import numpy as np
import os

# Function to calculate Intersection over Union (IoU)
def calculate_miou(pred, target, num_classes):
    """
    Calculate the mean Intersection over Union (mIoU) for a single batch.

    Args:
        pred (Tensor): Predicted segmentation map [batch_size, height, width].
        target (Tensor): Ground truth segmentation map [batch_size, height, width].
        num_classes (int): Number of classes.

    Returns:
        float: Mean IoU for the batch.
    """
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    iou_list = []

    for cls in range(num_classes):
        # Create binary masks for the current class
        pred_mask = (pred == cls)
        target_mask = (target == cls)

        # Calculate intersection and union
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()

        if union == 0:
            iou_list.append(float('nan'))  # Ignore classes not present in target or prediction
        else:
            iou_list.append(intersection / union)

    return np.nanmean(iou_list)  # Ignore NaN values when averaging

# Load PASCAL VOC 2012 dataset
def get_voc_dataloader(batch_size):
    """
    Load the PASCAL VOC 2012 validation dataset with necessary transformations.

    Args:
        batch_size (int): Batch size for DataLoader.

    Returns:
        DataLoader: Dataloader for the PASCAL VOC 2012 validation set.
    """
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images to 256x256
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    target_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),  # Resize targets
        transforms.PILToTensor()  # Convert to tensor
    ])

    dataset = VOCSegmentation(
        root='./data',
        year='2012',
        image_set='val',
        download=False,
        transform=transform,
        target_transform=target_transform
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return dataloader

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

def plot_segmentation(predictions, targets, num_classes=21, save_path=None, index=0):
    """
    Plots segmentation predictions and ground truth side-by-side.

    Args:
        predictions (torch.Tensor or np.ndarray): Predicted segmentation maps (HxW or BxHxW).
        targets (torch.Tensor or np.ndarray): Ground truth segmentation maps (HxW or BxHxW).
        num_classes (int): Number of segmentation classes (default: 21).
        save_path (str, optional): Path to save the plot. If None, the plot will be shown.
        index (int): Index of the batch item to visualize if batch dimension exists.
    """
    # Ensure predictions and targets are numpy arrays
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # Handle batch dimensions
    if predictions.ndim == 3:  # If batch dimension exists
        predictions = predictions[index]
    if targets.ndim == 3:  # If batch dimension exists
        targets = targets[index]

    # Define a colormap for 21 classes
    colors = plt.cm.get_cmap('tab20', num_classes)
    cmap = ListedColormap(colors(range(num_classes)))

    # Create the plot
    plt.figure(figsize=(10, 5))
    
    # Plot predictions
    plt.subplot(1, 2, 1)
    plt.imshow(predictions, cmap=cmap, interpolation='nearest', vmin=0, vmax=num_classes - 1)
    plt.colorbar(ticks=np.arange(0, num_classes), label='Classes', shrink=0.8)
    plt.title('Predictions')
    plt.axis('off')

    # Plot ground truth
    plt.subplot(1, 2, 2)
    plt.imshow(targets, cmap=cmap, interpolation='nearest', vmin=0, vmax=num_classes - 1)
    plt.colorbar(ticks=np.arange(0, num_classes), label='Classes', shrink=0.8)
    plt.title('Ground Truth')
    plt.axis('off')

    # Show or save the plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved at: {save_path}")
    else:
        plt.show()


# Test function for the custom lightweight model
def test_custom_model(model_path, batch_size=16, num_classes=21):
    """
    Test the custom lightweight segmentation model on the PASCAL VOC 2012 dataset.

    Args:
        model_path (str): Path to the trained model checkpoint.
        batch_size (int): Batch size for DataLoader.
        num_classes (int): Number of classes in the dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    print("Loading the trained custom model...")
    model = student(num_classes=num_classes).to(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Model loaded from {model_path}")

    # Load PASCAL VOC validation dataset
    dataloader = get_voc_dataloader(batch_size)
    print("Dataset loaded.")

    # Evaluate model
    miou_list = []
    print("Starting evaluation...")
    with torch.no_grad():
        for idx, (images, targets) in enumerate(dataloader):
            images, targets = images.to(device), targets.squeeze(1).to(device)  # Remove extra channel from targets

            # Get model predictions
            outputs = model(images)
            if isinstance(outputs, tuple):
                logits = outputs[0]  # Get the first output (logits)
            else:
                logits = outputs  # If not a tuple, assume it's the logits directly
            predictions = torch.argmax(logits, dim=1)  # Convert logits to class indices
            # Calculate mIoU for the batch
            batch_miou = calculate_miou(predictions, targets, num_classes)
            plot_segmentation(predictions, targets, 21, save_path=None)
            miou_list.append(batch_miou)

            if idx % 10 == 0:
                print(f"Processed {idx}/{len(dataloader)} batches, Current Batch mIoU: {batch_miou:.4f}")

    # Compute overall mIoU
    overall_miou = np.nanmean(miou_list)
    print(f"Mean Intersection over Union (mIoU): {overall_miou:.4f}")

if __name__ == "__main__":
    # good model is nstudent_32_0.0002_1e-05
    # distilled is student_distilled
    model_checkpoint = "./models/student_distilled.pth"  # Path to the trained custom model checkpoint
    test_custom_model(model_path=model_checkpoint)
