# from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, ConvTranspose2d, ReLU, MaxPool2d, Sigmoid, Parameter, Softmax, Upsample
# from torch import tensor, cat
# import torch.nn.functional as F
#
#
# class ConvBlock(Module):
#     def __init__(self, in_channel, out_channel, batch_norm_momentum):
#         super(ConvBlock, self).__init__()
#         self.in_channel = in_channel
#         self.out_channel = out_channel
#         self.batch_norm_momentum = batch_norm_momentum
#         self.conv1 = Conv2d(self.in_channel, self.out_channel, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
#         self.relu = ReLU()
#         self.batchnorm = BatchNorm2d(self.out_channel, momentum=self.batch_norm_momentum)
#         self.conv2 = Conv2d(self.out_channel, self.out_channel, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
#
#     def forward(self, x: tensor) -> tensor:
#         inputs = x
#         x_conv1 = self.conv1(inputs)
#         x_1 = self.relu(self.batchnorm(x_conv1))
#         x_conv2 = self.conv2(x_1)
#         x_2 = self.relu(self.batchnorm(x_conv2))
#         return x_2
#
#
# class UnConv(Module):
#     def __init__(self, in_channel, out_channel, batch_norm_momentum):
#         super(UnConv, self).__init__()
#         self.in_channel = in_channel
#         self.out_channel = out_channel
#         self.batch_norm_momentum = batch_norm_momentum
#
#         # self.unconv = ConvTranspose2d(self.in_channel, self.in_channel, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2), output_padding=(1, 1))
#         self.unconv = Upsample(scale_factor=2, mode='bilinear', align_corners=False)
#
#         self.conv = Conv2d(self.in_channel, self.out_channel, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
#         self.relu = ReLU()
#         self.batchnorm = BatchNorm2d(self.out_channel, momentum=self.batch_norm_momentum)
#
#     def forward(self, x: tensor) -> tensor:
#         inputs = x
#         # x_unconv = F.interpolate(inputs, scale_factor=2, mode='bilinear')
#         x_unconv = self.unconv(inputs)
#         x_out = self.relu(self.batchnorm(self.conv(x_unconv)))
#         return x_out
#
#
# class UNet(Module):
#
#     def __init__(self, num_classes: int = 10, iterations: int = 1, multiplier: float = 1.0, num_layers: int = 4, integrate: bool = False):
#         super(UNet, self).__init__()
#         self.num_classes = num_classes
#         self.iterations = iterations
#         self.multiplier = multiplier
#         self.num_layers = num_layers
#         self.integrate = integrate
#         self.batch_norm_momentum = 0.01
#
#         self.conv1 = ConvBlock(1, 32, self.batch_norm_momentum)
#         self.pool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
#
#         self.conv2 = ConvBlock(32, 64, self.batch_norm_momentum)
#         self.pool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
#
#         self.conv3 = ConvBlock(64, 128, self.batch_norm_momentum)
#         self.pool3 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
#
#         self.conv4 = ConvBlock(128, 256, self.batch_norm_momentum)
#         self.pool4 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
#
#         self.middle = ConvBlock(256, 512, self.batch_norm_momentum)
#
#         self.unconv1 = UnConv(512, 256, self.batch_norm_momentum)
#
#         self.up4 = ConvBlock(512, 256, self.batch_norm_momentum)
#
#         self.unconv2 = UnConv(256, 128, self.batch_norm_momentum)
#
#         self.up3 = ConvBlock(256, 128, self.batch_norm_momentum)
#
#         self.unconv3 = UnConv(128, 64, self.batch_norm_momentum)
#
#         self.up2 = ConvBlock(128, 64, self.batch_norm_momentum)
#
#         self.unconv4 = UnConv(64, 32, self.batch_norm_momentum)
#
#         self.up1 = ConvBlock(64, 32, self.batch_norm_momentum)
#
#         self.post = Conv2d(32, self.num_classes, kernel_size=(1, 1), stride=(1, 1))
#         self.output = Softmax(dim=1)
#
#     def forward(self, x: tensor) -> tensor:
#         inputs = x
#         conv1 = self.conv1(inputs)
#         pool1 = self.pool1(conv1)
#
#         conv2 = self.conv2(pool1)
#         pool2 = self.pool2(conv2)
#
#         conv3 = self.conv3(pool2)
#         pool3 = self.pool3(conv3)
#
#         conv4 = self.conv4(pool3)
#         pool4 = self.pool4(conv4)
#
#         middle = self.middle(pool4)
#
#         unconv1 = self.unconv1(middle)
#         # print(unconv1.shape)
#         # print(conv4.shape)
#         # exit()
#         up4 = self.up4(cat((conv4, unconv1), dim=1))
#
#         unconv2 = self.unconv2(up4)
#         up3 = self.up3(cat((conv3, unconv2), dim=1))
#
#         unconv3 = self.unconv3(up3)
#         up2 = self.up2(cat((conv2, unconv3), dim=1))
#
#         unconv4 = self.unconv4(up2)
#         up1 = self.up1(cat((conv1, unconv4), dim=1))
#
#         post = self.post(up1)
#         output = self.output(post)
#
#         return output


import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # print('x5', x1.shape)
        # print('x4', x2.shape)

        x1 = self.up(x1)
        # print('x1', x1.shape)
        # exit()
        # # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.conv(x))


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # 原版
        # self.inc = DoubleConv(n_channels, 64)
        # self.down1 = Down(64, 128)
        # self.down2 = Down(128, 256)
        # self.down3 = Down(256, 512)
        # factor = 2 if bilinear else 1
        # self.down4 = Down(512, 1024 // factor)  # 512
        # self.up1 = Up(1024, 512 // factor, bilinear)
        # self.up2 = Up(512, 256 // factor, bilinear)
        # self.up3 = Up(256, 128 // factor, bilinear)
        # self.up4 = Up(128, 64, bilinear)
        # self.outc = OutConv(64, n_classes)

        # upsample bilinear=True
        # self.inc = DoubleConv(n_channels, 32)
        # self.down1 = Down(32, 64)
        # self.down2 = Down(64, 128)
        # self.down3 = Down(128, 256)
        # factor = 2 if bilinear else 1
        # self.down4 = Down(256, 512 // factor)  # 512
        # self.up1 = Up(512, 256 // factor, bilinear)
        # self.up2 = Up(256, 128 // factor, bilinear)
        # self.up3 = Up(128, 64 // factor, bilinear)
        # self.up4 = Up(64, 32, bilinear)
        # self.outc = OutConv(32, n_classes)

        # unconv bilinear=False
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)  # 512
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # print(x5.shape)
        # exit()
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
