import torch
import torch.nn as nn

from torch.autograd import Variable

# (bs, 128, 128, 100, 1)


def conv_3d_block(in_channels, out_channels, kernels=3, padding=1, kernel_size=(2, 2, 2), stride=(2, 2, 2)):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=kernels, padding=padding),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(),
        nn.MaxPool3d(kernel_size, stride),
    )


def double_conv_3d_block(in_channels, out_channels, kernels=3, padding=1, kernel_size=(2, 2, 2), stride=(2, 2, 2)):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=kernels, padding=padding),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(),
        nn.Conv3d(out_channels, out_channels, kernel_size=kernels, padding=padding),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(),
        nn.MaxPool3d(kernel_size=kernel_size, stride=stride),
    )


class CNN(nn.Module):
    """
    Simple 3D Convolution Neural Network (CNN) with Batch Norm
    Modified from https://github.com/xmuyzz/3D-CNN-PyTorch/blob/master/models/C3DNet.py
    """

    def __init__(self, in_channels: int, out_channels: int, depth: int = 64):
        """
        Args:
            in_channels (int): The number of inputs
            out_channels (int): The number of output classes
            depth (int): The number of slices extracted from the CT Scan
        """
        super().__init__()
        self.conv_block_1 = conv_3d_block(in_channels, 64, 3, 1, (2, 2, 2), (1, 2, 2))
        self.conv_block_2 = conv_3d_block(64, 128, 3, 1, (2, 2, 2), (2, 2, 2))
        self.conv_block_3 = double_conv_3d_block(128, 256, 3, 1, (2, 2, 2), (2, 2, 2))
        self.conv_block_4 = double_conv_3d_block(256, 512, 3, 1, (2, 2, 2), (2, 2, 2))

        self.conv_block_5 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1)),
        )

        self.fc1 = nn.Sequential(
            nn.Linear((512 * ???), 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.fc3 = nn.Sequential(nn.Linear(4096, out_channels)) # TODO: how can we handle 

    def forward(self, x):
        out = self.conv_block_1(x)
        print(out.shape)
        out = self.conv_block_2(out)
        print(out.shape)
        out = self.conv_block_3(out)
        print(out.shape)
        out = self.conv_block_4(out)
        print(out.shape)
        out = self.conv_block_5(out)
        print(out.shape)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


if __name__ == "__main__":

    model = CNN(in_channels=1, out_channels=10)  # 10 classes
    print(model)

    input_var = Variable(torch.randn(32, 1, 100, 128, 128))
    output = model(input_var)
