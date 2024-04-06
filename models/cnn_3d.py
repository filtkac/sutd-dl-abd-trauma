import torch
import torch.nn as nn

from torch.autograd import Variable


def conv_3d_block(in_channels, out_channels, kernels=3, padding=1, pool_kernels=2, pool_stride=2):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=kernels, padding=padding),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(),
        nn.MaxPool3d(kernel_size=pool_kernels, stride=pool_stride),
    )


def double_conv_3d_block(
    in_channels, out_channels, kernels=3, padding=1, pool_kernels=2, pool_stride=2, pool_padding=0
):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=kernels, padding=padding),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(),
        nn.Conv3d(out_channels, out_channels, kernel_size=kernels, padding=padding),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(),
        nn.MaxPool3d(kernel_size=pool_kernels, stride=pool_stride, padding=pool_padding),
    )


class ConvNet3D(nn.Module):
    """
    Simple 3D Convolution Neural Network (CNN) with Batch Norm
    Modified from https://github.com/xmuyzz/3D-CNN-PyTorch/blob/master/models/C3DNet.py
    """

    def __init__(self, in_channels, out_channels, depth=64, height=128, width=128):
        """
        Args:
            in_channels (int): The number of input channels
            out_channels (int): The number of output classes
            depth (int): The number of slices extracted from the CT scan
            height (int): The height of the CT scan inputs
            width (int): The width of the CT scan inputs
        """
        super().__init__()

        # (N, C, D, H, W) -> (N, 1, 64, 128, 128)
        self.conv_block_1 = conv_3d_block(in_channels, 64)  # output: (N, 64, 32, 64, 64)
        self.conv_block_2 = conv_3d_block(64, 128)  # output: (N, 128, 16, 32, 32)
        self.conv_block_3 = double_conv_3d_block(128, 256)  # output: (N, 64, 8, 16, 16)
        self.conv_block_4 = double_conv_3d_block(256, 512)  # output: (N, 64, 4, 8, 8)
        self.conv_block_5 = double_conv_3d_block(512, 512)  # output: (N, 512, 2, 4, 4)

        fc_inputs = 512 * (depth // 2**5) * (height // 2**5) * (width // 2**5)  # 5 max pooling layers -> 2^5

        self.fc1 = nn.Sequential(
            nn.Linear(fc_inputs, 4096),  # output: (N, 4096)
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096),  # output: (N, 4096)
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.fc3 = nn.Linear(4096, out_channels)  # (N, output_channels)

    def forward(self, x):
        out = self.conv_block_1(x)
        out = self.conv_block_2(out)
        out = self.conv_block_3(out)
        out = self.conv_block_4(out)
        out = self.conv_block_5(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out  # no need to apply sigmoid since we are using BCEwithLogitsLoss


if __name__ == "__main__":
    batch_size, channels, depth, height, width = 32, 1, 64, 128, 128
    n_classes = 5
    model = ConvNet3D(
        in_channels=channels, out_channels=n_classes, depth=depth, height=height, width=width
    )  # 5 classes
    print(model)

    input_var = Variable(torch.randn(batch_size, channels, depth, height, width))
    output = model(input_var)
    assert output.shape == (batch_size, n_classes)
    print("Tested 3D CNN Model")
