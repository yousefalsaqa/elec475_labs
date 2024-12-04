import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import fcn_resnet50
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
from model import *  # Import your custom student model
import os
import matplotlib.pyplot as plt
import numpy as np

# Hyperparameters
ALPHA = 0.5  # Weight for ground truth loss
BETA = 0.5   # Weight for distillation loss
TAU = 2.0    # Temperature scaling for softmax
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
NUM_CLASSES = 21  # PASCAL VOC 2012 has 21 classes
NUM_EPOCHS = 20

# Define data transformations
def get_transforms():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    target_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor()
    ])
    return transform, target_transform

# Load datasets
def get_dataloaders(batch_size):
    transform, target_transform = get_transforms()

    train_dataset = VOCSegmentation(
        root='./data',
        year='2012',
        image_set='train',
        download=True,
        transform=transform,
        target_transform=target_transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    val_dataset = VOCSegmentation(
        root='./data',
        year='2012',
        image_set='val',
        download=True,
        transform=transform,
        target_transform=target_transform
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader

# Train for one epoch
def train_one_epoch(student_model, teacher_model, dataloader, optimizer, loss_fn, device):
    student_model.train()
    teacher_model.eval()  # Teacher is frozen
    total_loss = 0.0

    for imgs, targets in dataloader:
        imgs, targets = imgs.to(device), targets.squeeze(1).to(device, dtype=torch.long)

        # Forward pass through student and teacher
        student_outputs, _ = student_model(imgs)
        with torch.no_grad():
            teacher_outputs = teacher_model(imgs)['out']

        # Response-based loss
        student_logits = F.log_softmax(student_outputs / TAU, dim=1)
        teacher_logits = F.softmax(teacher_outputs / TAU, dim=1)
        # distillation_loss = F.kl_div(student_logits, teacher_logits, reduction='batchmean') * (TAU ** 2)
        distillation_loss = F.cross_entropy(student_logits, teacher_logits
                                            , weight=None)
        # Ground truth loss
        ground_truth_loss = loss_fn(student_outputs, targets)

        # Combined loss
        total_loss_value = ALPHA * ground_truth_loss + BETA * distillation_loss

        # Backpropagation
        optimizer.zero_grad()
        total_loss_value.backward()
        optimizer.step()

        total_loss += total_loss_value.item()

    return total_loss / len(dataloader)

# Validation step
def validate(student_model, val_loader, loss_fn, device):
    student_model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs, targets = imgs.to(device), targets.squeeze(1).to(device, dtype=torch.long)
            outputs, _ = student_model(imgs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()

    return total_loss / len(val_loader)

# Plot loss curves
def plot_losses(train_losses, val_losses, save_path="./loss/response_loss_curve.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Distillation Loss Curve")
    plt.legend()
    plt.savefig(save_path)
    # plt.show()

# Main function
def main():
    # Set device
    model_path = "./models/nstudent_32_0.0002_1e-05.pth"  # Path to the trained custom model checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load teacher model (pretrained FCN-ResNet50)
    teacher_model = fcn_resnet50(pretrained=True).to(device)
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Load student model
    student_model = student(num_classes=NUM_CLASSES).to(device)
    student_model.load_state_dict(torch.load(model_path))

    # Load data
    train_loader, val_loader = get_dataloaders(BATCH_SIZE)

    # Loss function and optimizer
    weights = 1/(np.array([1400, 108, 94, 137, 124, 195, 121, 209, 154, 303, 152, 86, 149, 100, 101, 868, 151, 155, 103, 96, 101]))
    weights = weights / weights.sum()
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)  # Ignore void class
    optimizer = torch.optim.Adam(student_model.parameters(), lr=LEARNING_RATE)

    # Training loop
    train_losses, val_losses = [], []

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        # Train for one epoch
        train_loss = train_one_epoch(student_model, teacher_model, train_loader, optimizer, loss_fn, device)
        train_losses.append(train_loss)

        # Validate
        val_loss = validate(student_model, val_loader, loss_fn, device)
        val_losses.append(val_loss)
        
        plot_losses(train_losses, val_losses)
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save the distilled student model
        if not os.path.exists("./models"):
            os.makedirs("./models")
        torch.save(student_model.state_dict(), "./models/response_distilled.pth")
        print("Distilled student model saved.")

    # Plot loss curves
    print("Distillation training complete.")

if __name__ == "__main__":
    main()
