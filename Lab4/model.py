import torch
import torch.nn as nn
import torch.nn.functional as F

<<<<<<< HEAD:Lab4/lightweight_model.py
class OptimizedSegmentationModel(nn.Module):
    def __init__(self, num_classes=21):
        super(OptimizedSegmentationModel, self).__init__()

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample
        )
=======
class student(nn.Module):
    def __init__(self, num_classes=21):
        super(student, self).__init__()
>>>>>>> 10495f12a424982a27deedc9f2d999a741085b9d:Lab4/model.py

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Decoder
        self.decoder3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

        # Output
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        bottleneck = self.bottleneck(enc3)

        dec3 = self.decoder3(bottleneck)
        enc3_resized = F.interpolate(enc3, size=dec3.shape[2:], mode='bilinear', align_corners=False)
        dec2 = self.decoder2(dec3 + enc3_resized)

        enc2_resized = F.interpolate(enc2, size=dec2.shape[2:], mode='bilinear', align_corners=False)
        dec1 = self.decoder1(dec2 + enc2_resized)

        enc1_resized = F.interpolate(enc1, size=dec1.shape[2:], mode='bilinear', align_corners=False)
        out = self.final(dec1 + enc1_resized)
        return out


<<<<<<< HEAD:Lab4/lightweight_model.py
# Test the model (optional)
if __name__ == "__main__":
    model = OptimizedSegmentationModel(num_classes=21)
    input_tensor = torch.randn(1, 3, 128, 128)
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
=======
def get_model_size(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())  # Parameters size
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())    # Buffers size
    total_size = param_size + buffer_size  # Total size in bytes
    return total_size

# Test the model
if __name__ == "__main__":
    model = student(num_classes=21)
    input_tensor = torch.randn(1, 3, 256, 256)  # Example input (batch_size, channels, height, width)
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")  # Should match (batch_size, num_classes, height, width)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Total size: {get_model_size(model)}")

>>>>>>> 10495f12a424982a27deedc9f2d999a741085b9d:Lab4/model.py
