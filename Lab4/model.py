import torch
import torch.nn as nn
import torch.nn.functional as F

class student(nn.Module):
    def __init__(self, num_classes=21):
        super(student, self).__init__()

        # Encoder with depthwise separable convolutions
        self.encoder1 = self._depthwise_separable_conv(3, 32)  # Input to 32 channels
        self.encoder2 = self._depthwise_separable_conv(32, 64)  # Downsample to 64 channels
        self.encoder3 = self._depthwise_separable_conv(64, 128)  # Downsample to 128 channels
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Decoder with bilinear upsampling + convolution for checkerboard artifact avoidance
        self.decoder1 = self._upsample_conv(256, 128)  # Upsample to 128 channels
        self.decoder2 = self._upsample_conv(256, 64)  # Skip connection from encoder3
        self.decoder3 = self._upsample_conv(128, 32)  # Skip connection from encoder2

        # Boundary refinement block
        self.boundary_refinement = nn.Sequential(
            nn.Conv2d(32 + 32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # Final classifier
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

    def _depthwise_separable_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),  # Depthwise
            nn.Conv2d(in_channels, out_channels, kernel_size=1),  # Pointwise
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def _upsample_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # Bilinear upsampling
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)  # Output: 32 channels
        enc2 = self.encoder2(enc1)  # Output: 64 channels
        enc3 = self.encoder3(enc2)  # Output: 128 channels

        # Bottleneck
        bottleneck_out = self.bottleneck(enc3)  # Output: 256 channels

        # Decoder with skip connections
        dec1 = self.decoder1(bottleneck_out)  # Output: 128 channels
        enc3_resized = F.interpolate(enc3, size=dec1.shape[2:], mode='bilinear', align_corners=True)
        dec2 = self.decoder2(torch.cat([dec1, enc3_resized], dim=1))  # Output: 64 channels
        enc2_resized = F.interpolate(enc2, size=dec2.shape[2:], mode='bilinear', align_corners=True)
        dec3 = self.decoder3(torch.cat([dec2, enc2_resized], dim=1))  # Output: 32 channels

        # Boundary refinement with skip connection from encoder1
        enc1_resized = F.interpolate(enc1, size=dec3.shape[2:], mode='bilinear', align_corners=True)
        refined = self.boundary_refinement(torch.cat([dec3, enc1_resized], dim=1))

        # Final classifier
        output = self.classifier(refined)
        output = F.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=True)

        return output


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

