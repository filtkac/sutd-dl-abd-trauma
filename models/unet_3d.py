import torch
from torch import nn
from torch import relu

from torch.autograd import Variable


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, depth=64, height=128, width=128):
        super().__init__()

        # Encoder
        # In the encoder, convolutional layers with the Conv3d function are used to extract features from the input image.
        # -------
        # input: 1 channel 64x128x128
        self.e11 = nn.Conv3d(in_channels, 64, kernel_size=3, padding=1)
        self.e12 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)  # output: 64 channels 32x64x64

        # input: 64 channels 32x64x64
        self.e21 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.e22 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)  # output: 128 channels 16x32x32

        # input: 128 channels 16x32x32
        self.e31 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.e32 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)  # output: 256 channels 8x16x16

        # input: 256 channels 8x16x16
        self.e41 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.e42 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)  # output: 512 channels 4x8x8

        # input: 512 channels 4x8x8
        self.e51 = nn.Conv3d(512, 1024, kernel_size=3, padding=1)
        self.e52 = nn.Conv3d(1024, 1024, kernel_size=3, padding=1)  # output 1024 channels 4x8x8

        # Decoder
        self.upconv1 = nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv3d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv3d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv3d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv3d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv3d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        # back to 64 * 128 * 128

        # need to reduce this, otherwise getting memory errors for the linear layer, too many parameters?
        self.pool5 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.pool6 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.pool7 = nn.MaxPool3d(kernel_size=2, stride=2)
        # 8 * 16 * 16

        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Sequential(
            nn.Linear(64 * (depth // 8) * (height // 8) * (width // 8), 4096),
            nn.ReLU(),
            nn.Dropout()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(4096, 128),
            nn.ReLU(),
            nn.Dropout()
        )

        self.fc3 = nn.Linear(128, out_channels)

    def forward(self, x):
        # Encoder
        xe11 = relu(self.e11(x))
        xe12 = relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = relu(self.e21(xp1))
        xe22 = relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = relu(self.e31(xp2))
        xe32 = relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = relu(self.e41(xp3))
        xe42 = relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = relu(self.e51(xp4))
        xe52 = relu(self.e52(xe51))

        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = relu(self.d11(xu11))
        xd12 = relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = relu(self.d21(xu22))
        xd22 = relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = relu(self.d31(xu33))
        xd32 = relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = relu(self.d41(xu44))
        xd42 = relu(self.d42(xd41))

        # Output layer
        out = self.pool5(xd42)
        out = self.pool6(out)
        out = self.pool7(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out


if __name__ == "__main__":
    batch_size, channels, depth, height, width = 2, 1, 64, 128, 128
    n_classes = 5
    model = UNet(
        in_channels=channels,
        out_channels=n_classes,
        depth=depth,
        height=height,
        width=width
    )  # 5 classes
    print(model)

    input_var = Variable(torch.randn(batch_size, channels, depth, height, width))
    output = model(input_var)
    assert output.shape == (batch_size, n_classes)
    print("Tested UNet Model")
