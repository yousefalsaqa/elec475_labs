import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
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

# Test function for FCN-ResNet50
def test_fcn_resnet50(batch_size=16, num_classes=21):
    """
    Test the pretrained FCN-ResNet50 model on the PASCAL VOC 2012 dataset.

    Args:
        batch_size (int): Batch size for DataLoader.
        num_classes (int): Number of classes in the dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained model
    print("Loading pretrained FCN-ResNet50 model...")
    model = torchvision.models.segmentation.fcn_resnet50(pretrained=True).to(device)
    model.eval()

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
            outputs = model(images)['out']
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
    test_fcn_resnet50()
