import torch
import torch.nn as nn

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, init_features=32):
        super(UNet3D, self).__init__()
        features = init_features
        self.encoder1 = self.block(in_channels, features)
        self.pool1 = nn.MaxPool3d(2)
        self.encoder2 = self.block(features, features*2)
        self.pool2 = nn.MaxPool3d(2)

        self.bottleneck = self.block(features*2, features*4)

        self.up2 = nn.ConvTranspose3d(features*4, features*2, 2, 2)
        self.decoder2 = self.block(features*4, features*2)
        self.up1 = nn.ConvTranspose3d(features*2, features, 2, 2)
        self.decoder1 = self.block(features*2, features)

        self.output = nn.Conv3d(features, out_channels, 1)

    def block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.up2(b)
        d2 = self.decoder2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.decoder1(torch.cat([d1, e1], dim=1))
        return self.output(d1)
