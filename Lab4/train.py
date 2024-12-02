import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import student
from torchvision.transforms.functional import InterpolationMode


class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, device, num_epochs, model_name, lr, batch_size):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.model_name = model_name
        self.lr = lr
        self.batch_size = batch_size
        self.train_losses = []
        self.val_losses = []

        # Initialize live plotting
        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        plt.ion()

    def compute_loss(self, outputs, targets):
        loss_fn = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore -1 (background/ignore class)
        return loss_fn(outputs, targets)

    def save_plot(self):
        plot_filename = f"./loss/loss_{self.model_name}_lr{self.lr}_bs{self.batch_size}_curve.png"
        plt.savefig(plot_filename, dpi=300)
        print(f"Plot saved at {plot_filename}")

    def save_model(self):
        model_filename = f"./models/model_{self.model_name}_lr{self.lr}_bs{self.batch_size}.pth"
        torch.save(self.model.state_dict(), model_filename)
        print(f"Model saved at {model_filename}")

    def plot_loss(self):
        self.ax.clear()
        self.ax.plot(self.train_losses, label="Train Loss")
        self.ax.plot(self.val_losses, label="Validation Loss")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.ax.set_title("Training and Validation Loss")
        self.ax.legend()
        plt.grid()
        plt.draw()
        plt.pause(0.1)

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        for images, masks in tqdm(self.train_loader, desc="Training"):
            images, masks = images.to(self.device), masks.to(self.device)
            self.optimizer.zero_grad()
            predictions, _ = self.model(images)  # Forward pass
            loss = self.compute_loss(predictions, masks)
            loss.backward()  # Backpropagation
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def validate_epoch(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc="Validating"):
                images, masks = images.to(self.device), masks.to(self.device)
                predictions, _ = self.model(images)  # Forward pass
                loss = self.compute_loss(predictions, masks)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def run(self):
        try:
            for epoch in range(1, self.num_epochs + 1):
                print(f"Epoch {epoch}/{self.num_epochs}")
                train_loss = self.train_epoch()
                val_loss = self.validate_epoch()
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
                self.plot_loss()
        except KeyboardInterrupt:
            print("Training interrupted. Saving progress...")
        finally:
            self.save_model()
            self.save_plot()
            plt.ioff()
            plt.close(self.fig)

# Custom Transform Class
class Replace255WithNeg1:
    def __call__(self, tensor):
        tensor = tensor.squeeze(0).long()
        tensor[tensor == 255] = -1
        return tensor

def prepare_transforms():
    """Returns image and mask transforms for training and validation."""
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    mask_transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST),
        transforms.PILToTensor(),
        Replace255WithNeg1(),  # Use the custom transform class
    ])

    return img_transform, mask_transform_pipeline



def prepare_dataloader(root_path, batch_size):
    """Creates DataLoader for Pascal VOC."""
    img_transform, mask_transform = prepare_transforms()

    train_data = VOCSegmentation(
        root=root_path,
        year="2012",
        image_set="train",
        download=False,
        transform=img_transform,
        target_transform=mask_transform,
    )

    val_data = VOCSegmentation(
        root=root_path,
        year="2012",
        image_set="val",
        download=False,
        transform=img_transform,
        target_transform=mask_transform,
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader


def main():
    # Parameters
    batch_size = 16
    learning_rate = 0.002
    num_epochs = 25
    root_path = "./Lab4/data/"

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare dataloaders
    train_loader, val_loader = prepare_dataloader(root_path, batch_size)

    # Initialize model and optimizer
    model = student(num_classes=21).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        model_name="student",
        lr=learning_rate,
        batch_size=batch_size,
    )

    # Run training
    trainer.run()


if __name__ == "__main__":
    main()
