import torch
import torch.nn as nn
import torch.nn.functional as F

class student(nn.Module):
    def __init__(self, in_channels=3, num_classes=21):
        super(student, self).__init__()

        # Encoder
        self.enc1 = self.double_conv(in_channels, 32)  # Reduced from 64 to 32
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = self.double_conv(32, 64)  # Reduced from 128 to 64
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = self.double_conv(64, 128)  # Reduced from 256 to 128
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = self.double_conv(128, 256)  # Reduced from 512 to 256
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = self.double_conv(256, 512)  # Reduced from 1024 to 512

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self.double_conv(512, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self.double_conv(256, 128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self.double_conv(128, 64)

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self.double_conv(64, 32)

        # Output layer
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def double_conv(self, in_channels, out_channels):
        """
        Helper function for a double convolution block:
        Conv2D -> ReLU -> Conv2D -> ReLU
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def crop_tensor(self, encoder_tensor, target_tensor):
        """
        Crop the encoder tensor to match the spatial dimensions of the target tensor.
        """
        _, _, h, w = target_tensor.size()
        enc_h, enc_w = encoder_tensor.size(2), encoder_tensor.size(3)
        delta_h, delta_w = enc_h - h, enc_w - w
        top_crop = delta_h // 2
        left_crop = delta_w // 2
        return encoder_tensor[:, :, top_crop:top_crop + h, left_crop:left_crop + w]

    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))  # Intermediate feature map
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder path
        up4 = self.upconv4(bottleneck)
        cropped_enc4 = self.crop_tensor(enc4, up4)
        dec4 = self.dec4(torch.cat([up4, cropped_enc4], dim=1))

        up3 = self.upconv3(dec4)
        cropped_enc3 = self.crop_tensor(enc3, up3)
        dec3 = self.dec3(torch.cat([up3, cropped_enc3], dim=1))

        up2 = self.upconv2(dec3)
        cropped_enc2 = self.crop_tensor(enc2, up2)
        dec2 = self.dec2(torch.cat([up2, cropped_enc2], dim=1))

        up1 = self.upconv1(dec2)
        cropped_enc1 = self.crop_tensor(enc1, up1)
        dec1 = self.dec1(torch.cat([up1, cropped_enc1], dim=1))

        # Output
        out = self.final_conv(dec1)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out, enc2  # Return final output and intermediate feature map



# Test the model
if __name__ == "__main__":
    model = student(in_channels=3, num_classes=21)
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    print(f"Output shape: {output[0].shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
