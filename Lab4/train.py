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


# Define Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.softmax(preds, dim=1)  # Apply softmax to predictions

        # Validate and preprocess targets
        valid_mask = (targets >= 0) & (targets < preds.shape[1])
        targets = torch.where(valid_mask, targets, torch.zeros_like(targets))
        targets_one_hot = nn.functional.one_hot(targets, num_classes=preds.shape[1]).permute(0, 3, 1, 2).float()

        intersection = torch.sum(preds * targets_one_hot, dim=(2, 3))
        union = torch.sum(preds, dim=(2, 3)) + torch.sum(targets_one_hot, dim=(2, 3))

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_score.mean()


class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, device, num_epochs, model_name, lr, batch_size):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.model_name = model_name
        self.lr = lr
        self.batch_size = batch_size
        self.train_losses = []
        self.val_losses = []
        self.loss_fn = DiceLoss()

        # Visualization
        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        plt.ion()

    def unnormalize(self, images, mean, std):
        """
        Unnormalize a batch of images.
        """
        mean = torch.tensor(mean).view(1, 3, 1, 1).to(images.device)  # Broadcast mean across batch
        std = torch.tensor(std).view(1, 3, 1, 1).to(images.device)    # Broadcast std across batch
        return images * std + mean


    def compute_loss(self, outputs, targets):
        try:
            loss = self.loss_fn(outputs, targets)
            return loss
        except Exception as e:
            print(f"Error in loss computation: {e}")
            raise

    def save_plot(self):
        plot_filename = f"./loss/loss_{self.model_name}_lr{self.lr}_bs{self.batch_size}_curve.png"
        plt.savefig(plot_filename, dpi=300)
        print(f"Plot saved at {plot_filename}")

    def save_model(self, epoch=None):
        model_filename = f"./models/model_{self.model_name}_lr{self.lr}_bs{self.batch_size}_{'epoch' + str(epoch) if epoch else 'final'}.pth"
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

    def visualize_batch(self, images, predictions, masks, mean, std):
        plt.figure(figsize=(15, 10))
        for i in range(min(len(images), 4)):  # Show a maximum of 4 samples
            unnormalized_image = self.unnormalize(images[i], mean, std).permute(1, 2, 0).cpu().numpy()
            plt.subplot(4, 3, 3 * i + 1)
            plt.title("Image")
            plt.imshow(unnormalized_image)
            plt.axis('off')

            plt.subplot(4, 3, 3 * i + 2)
            plt.title("Ground Truth")
            plt.imshow(masks[i].cpu())
            plt.axis('off')

            plt.subplot(4, 3, 3 * i + 3)
            plt.title("Prediction")
            plt.imshow(torch.argmax(predictions, dim=1)[i].cpu())
            plt.axis('off')
        plt.show()

    def train_epoch(self, mean, std):
        self.model.train()
        total_loss = 0.0
        for images, masks in tqdm(self.train_loader, desc="Training"):
            images, masks = images.to(self.device), masks.to(self.device)

            self.optimizer.zero_grad()
            predictions, _ = self.model(images)  # Forward pass

            # Debug predictions
            unique_preds = torch.unique(torch.argmax(predictions, dim=1))
            print(f"Unique Predictions in Batch: {unique_preds}")

            loss = self.compute_loss(predictions, masks)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            # Visualize occasionally
            if torch.rand(1).item() < 0.1:
                self.visualize_batch(images, predictions, masks, mean, std)

        return total_loss / len(self.train_loader)

    def validate_epoch(self, mean, std):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc="Validating"):
                images, masks = images.to(self.device), masks.to(self.device)
                predictions, _ = self.model(images)

                # Debug predictions
                unique_preds = torch.unique(torch.argmax(predictions, dim=1))
                print(f"Validation Unique Predictions: {unique_preds}")

                loss = self.compute_loss(predictions, masks)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def run(self, mean, std):
        try:
            for epoch in range(1, self.num_epochs + 1):
                print(f"Epoch {epoch}/{self.num_epochs}")
                train_loss = self.train_epoch(mean, std)
                val_loss = self.validate_epoch(mean, std)
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
                self.plot_loss()

                if self.scheduler is not None:
                    self.scheduler.step()

                self.save_model(epoch=epoch)
        except KeyboardInterrupt:
            print("Training interrupted. Saving progress...")
        finally:
            self.save_model()
            self.save_plot()
            plt.ioff()
            plt.close(self.fig)


def prepare_transforms():
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    mask_transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST),
        transforms.PILToTensor(),
        Replace255WithNeg1(),
    ])

    return img_transform, mask_transform_pipeline, mean, std


def prepare_dataloader(root_path, batch_size):
    img_transform, mask_transform, mean, std = prepare_transforms()

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

    return train_loader, val_loader, mean, std


class Replace255WithNeg1:
    def __call__(self, tensor):
        if tensor.ndimension() == 3 and tensor.shape[0] == 3:
            tensor = tensor[0]
        if tensor.ndimension() == 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        tensor = tensor.long()
        tensor[tensor == 255] = -1
        return tensor


def main():
    batch_size = 16
    learning_rate = 0.001
    num_epochs = 20
    root_path = "./Lab4/data/"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, mean, std = prepare_dataloader(root_path, batch_size)

    model = student(num_classes=21).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=num_epochs,
        model_name="student",
        lr=learning_rate,
        batch_size=batch_size,
    )

    trainer.run(mean, std)


if __name__ == "__main__":
    main()
