import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
from lightweight_model import LightweightSegmentationModel  # Import your custom model
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
        download=True,
        transform=transform,
        target_transform=target_transform
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return dataloader

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
    model = LightweightSegmentationModel(num_classes=num_classes).to(device)
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
            predictions = torch.argmax(outputs, dim=1)  # Convert logits to class indices

            # Calculate mIoU for the batch
            batch_miou = calculate_miou(predictions, targets, num_classes)
            miou_list.append(batch_miou)

            if idx % 10 == 0:
                print(f"Processed {idx}/{len(dataloader)} batches, Current Batch mIoU: {batch_miou:.4f}")

    # Compute overall mIoU
    overall_miou = np.nanmean(miou_list)
    print(f"Mean Intersection over Union (mIoU): {overall_miou:.4f}")

if __name__ == "__main__":
    model_checkpoint = "./models/lightweight_segmentation.pth"  # Path to the trained custom model checkpoint
    test_custom_model(model_path=model_checkpoint)
