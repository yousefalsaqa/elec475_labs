import torch
import torch.nn as nn
import torch.nn.functional as F

class student(nn.Module):
    def __init__(self, num_classes=21):
        super(student, self).__init__()

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


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Contracting Path (Encoder)
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        self.encoder5 = self.conv_block(512, 1024)

        # Expanding Path (Decoder)
        self.decoder4 = self.upconv_block(1024, 512)
        self.decoder3 = self.upconv_block(512, 256)
        self.decoder2 = self.upconv_block(256, 128)
        self.decoder1 = self.upconv_block(128, 64)

        # Final Convolution layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        """Convolution Block (Conv + ReLU + MaxPool)"""
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        return block

    def upconv_block(self, in_channels, out_channels):
        """Up-convolution Block (Upsample + Conv + ReLU)"""
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        return block

    def forward(self, x):
        # Contracting path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)

        # Expanding path with skip connections
        dec4 = self.decoder4(enc5)
        dec4 = torch.cat([dec4, enc4], dim=1)  # Concatenate encoder and decoder features
        dec4 = self.conv_block_1x1(dec4, 512)  # Reduce channels before next decoder layer
        
        dec3 = self.decoder3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.conv_block_1x1(dec3, 256)  # Reduce channels

        dec2 = self.decoder2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.conv_block_1x1(dec2, 128)  # Reduce channels

        dec1 = self.decoder1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.conv_block_1x1(dec1, 64)  # Reduce channels

        # Final output
        output = self.final_conv(dec1)
        return output

    def conv_block_1x1(self, x, out_channels):
        """1x1 Convolution to reduce the number of channels after concatenation"""
        return nn.Conv2d(x.size(1), out_channels, kernel_size=1)(x)


# Test the model (optional)
if __name__ == "__main__":
    # model = student(num_classes=21)
    model = UNet(in_channels=3, out_channels=21)
    input_tensor = torch.randn(1, 3, 256, 256)
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
