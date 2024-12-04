# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader, random_split
from torchsummary import summary
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import datetime
import argparse

batch_size=64

# %%
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

VOC_COLORMAP = np.array([
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128],
    ], dtype=np.uint8)

# Create a LUT to map RGB values to class indices
VOC_LUT = {tuple(color): idx for idx, color in enumerate(VOC_COLORMAP)}

# Function to convert RGB mask to class index mask
def rgb_to_class_index(mask):
    # Ensure mask is in the right format
    mask = np.array(mask, dtype=np.uint8)
    class_index_mask = np.zeros(mask.shape[:2], dtype=np.int64)

    # Map each pixel to its corresponding class index
    for rgb, class_idx in VOC_LUT.items():
        class_index_mask[np.all(mask == rgb, axis=-1)] = class_idx
    return class_index_mask

class FixMask:
    def __call__(self, img):
        class_index_mask = rgb_to_class_index(img)
        # Resize using nearest neighbor interpolation
        resize_transform = transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST)
        class_index_mask = Image.fromarray(class_index_mask.astype(np.uint8))
        resized_mask = resize_transform(class_index_mask)
    
        # Convert back to tensor
        return torch.tensor(np.array(resized_mask), dtype=torch.long)

# Define a named function to replace the lambda
def convert_to_tensor(mask):
    return torch.as_tensor(np.array(mask), dtype=torch.long)


target_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
    # FixMask(),
    # transforms.PILToTensor()
    transforms.Lambda(convert_to_tensor)  # Convert to tensor
])
# %%
train_set = VOCSegmentation(root='./data', 
                            year='2012',
                            image_set='train',
                            download=True,
                            transform=train_transform,
                            target_transform=target_transform)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
val_set = VOCSegmentation(root='./data', 
                        year='2012',
                        image_set='trainval',
                        download=True,
                        transform=val_transform,
                        target_transform=target_transform)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)


# %%
