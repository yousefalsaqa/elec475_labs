import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
from model import student
import matplotlib.pyplot as plt
from albumentations import (
    Compose, HorizontalFlip, RandomBrightnessContrast, ShiftScaleRotate
)
import albumentations.pytorch as alb
from torch.utils.data import WeightedRandomSampler




# Dice Loss with Weighted CrossEntropy
class WeightedDiceCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights, smooth=1e-6):
        super(WeightedDiceCrossEntropyLoss, self).__init__()
        self.class_weights = class_weights
        self.cross_entropy = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.softmax(preds, dim=1)

        # One-hot encoding for dice loss
        valid_mask = (targets >= 0) & (targets < preds.shape[1])
        targets = torch.where(valid_mask, targets, torch.zeros_like(targets))
        targets_one_hot = nn.functional.one_hot(targets, num_classes=preds.shape[1]).permute(0, 3, 1, 2).float()

        intersection = torch.sum(preds * targets_one_hot, dim=(2, 3))
        union = torch.sum(preds, dim=(2, 3)) + torch.sum(targets_one_hot, dim=(2, 3))
        dice_loss = 1 - (2 * intersection + self.smooth) / (union + self.smooth)

        # Combine CrossEntropy and Dice
        ce_loss = self.cross_entropy(preds, targets)
        return dice_loss.mean() + ce_loss

def get_augmentations():
    return Compose([
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        RandomBrightnessContrast(p=0.5),
        alb.ToTensorV2()
    ])

# Trainer Class
class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, device, num_epochs, class_weights):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.loss_fn = WeightedDiceCrossEntropyLoss(class_weights)

    def plot_confusion_matrix(self, preds, targets, num_classes):
        preds_flat = preds.cpu().numpy().flatten()
        targets_flat = targets.cpu().numpy().flatten()
        cm = confusion_matrix(targets_flat, preds_flat, labels=list(range(num_classes)))
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        for images, masks in tqdm(self.train_loader, desc="Training"):
            images, masks = images.to(self.device), masks.to(self.device, dtype=torch.long)
            self.optimizer.zero_grad()
            predictions, _ = self.model(images)
            loss = self.loss_fn(predictions, masks)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            print(f"Raw logits for a batch: {predictions[0].cpu().detach().numpy()}")


        return total_loss / len(self.train_loader)

    def validate_epoch(self):
        self.model.eval()
        total_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc="Validating"):
                images, masks = images.to(self.device), masks.to(self.device, dtype=torch.long)
                predictions, _ = self.model(images)
                loss = self.loss_fn(predictions, masks)
                total_loss += loss.item()

                all_preds.append(torch.argmax(predictions, dim=1).cpu())
                all_targets.append(masks.cpu())

        # Plot confusion matrix
        self.plot_confusion_matrix(torch.cat(all_preds), torch.cat(all_targets), num_classes=21)

        return total_loss / len(self.val_loader)

    def train(self):
        train_losses, val_losses = [], []
        best_val_loss = float('inf')

        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}\n")
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

            if self.scheduler:
                self.scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), "best_model.pth")

            # Plot Loss
            plt.figure()
            plt.plot(train_losses, label="Train Loss")
            plt.plot(val_losses, label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.title("Loss Curve")
            plt.savefig("loss_curve.png")
            plt.close()


def calculate_class_weights():
    object_counts = np.array([3507, 108, 94, 137, 124, 195, 121, 209, 154, 303, 152, 86, 149, 100, 101, 868, 151, 155, 103, 96, 101])
    weights = 1 / object_counts  # Inverse frequency
    weights[0] *= 0.5  # Penalize background less
    weights /= weights.sum()  # Normalize
    return torch.tensor(weights, dtype=torch.float32)

def get_balanced_sampler(dataset, class_counts):
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[target] for target in dataset.targets]
    sampler = WeightedRandomSampler(sample_weights, len(dataset))
    return sampler



def convert_to_tensor(mask):
    return torch.as_tensor(np.array(mask), dtype=torch.long)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Hyperparameters
    num_epochs = 20
    batch_size = 16
    learning_rate = 1e-3
    weight_decay = 1e-5

    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    target_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.Lambda(convert_to_tensor)
    ])

    train_set = VOCSegmentation(root="./data", year="2012", image_set="train", transform=transform, target_transform=target_transform)
    val_set = VOCSegmentation(root="./data", year="2012", image_set="val", transform=transform, target_transform=target_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model, Optimizer, and Scheduler
    model = student(in_channels=3, num_classes=21).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3)
    class_weights = calculate_class_weights().to(device)

    # Trainer
    trainer = Trainer(model, train_loader, val_loader, optimizer, scheduler, device, num_epochs, class_weights)
    trainer.train()


if __name__ == "__main__":
    main()
