from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, ConvTranspose2d, ReLU, MaxPool2d, Sigmoid, Parameter, Softmax, Upsample
from torch import tensor, cat


class ConvBlock(Module):
    def __init__(self, in_channel, out_channel, batch_norm_momentum):
        super(ConvBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.batch_norm_momentum = batch_norm_momentum
        self.conv1 = Conv2d(self.in_channel, self.out_channel, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.relu = ReLU()
        self.batchnorm = BatchNorm2d(self.out_channel, momentum=self.batch_norm_momentum)
        self.conv2 = Conv2d(self.out_channel, self.out_channel, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))

    def forward(self, x):
        inputs = x
        x_conv1 = self.conv1(inputs)
        x_1 = self.batchnorm(self.relu(x_conv1))
        x_conv2 = self.conv2(x_1)
        x_2 = self.batchnorm(self.relu(x_conv2))
        return x_2


class UnConv(Module):
    def __init__(self, in_channel, out_channel, batch_norm_momentum):
        super(UnConv, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.batch_norm_momentum = batch_norm_momentum

        # self.unconv = ConvTranspose2d(self.in_channel, self.in_channel, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2), output_padding=(1, 1))
        self.unconv = Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = Conv2d(self.in_channel, self.out_channel, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.relu = ReLU()
        self.batchnorm = BatchNorm2d(self.out_channel, momentum=self.batch_norm_momentum)

    def forward(self, x):
        inputs = x
        x_unconv = self.unconv(inputs)
        x_out = self.batchnorm(self.relu(self.conv(x_unconv)))
        return x_out


class UNet(Module):

    def __init__(self, num_classes: int = 10, iterations: int = 1, multiplier: float = 1.0, num_layers: int = 4, integrate: bool = False):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        self.iterations = iterations
        self.multiplier = multiplier
        self.num_layers = num_layers
        self.integrate = integrate
        self.batch_norm_momentum = 0.01

        self.conv1 = ConvBlock(1, 32, self.batch_norm_momentum)
        self.pool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))

        self.conv2 = ConvBlock(32, 64, self.batch_norm_momentum)
        self.pool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))

        self.conv3 = ConvBlock(64, 128, self.batch_norm_momentum)
        self.pool3 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))

        self.conv4 = ConvBlock(128, 256, self.batch_norm_momentum)
        self.pool4 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))

        self.middle = ConvBlock(256, 512, self.batch_norm_momentum)

        self.unconv1 = UnConv(512, 256, self.batch_norm_momentum)

        self.up4 = ConvBlock(512, 256, self.batch_norm_momentum)

        self.unconv2 = UnConv(256, 128, self.batch_norm_momentum)

        self.up3 = ConvBlock(256, 128, self.batch_norm_momentum)

        self.unconv3 = UnConv(128, 64, self.batch_norm_momentum)

        self.up2 = ConvBlock(128, 64, self.batch_norm_momentum)

        self.unconv4 = UnConv(64, 32, self.batch_norm_momentum)

        self.up1 = ConvBlock(64, 32, self.batch_norm_momentum)

        self.post = Conv2d(32, self.num_classes, kernel_size=(1, 1), stride=(1, 1))
        self.output = Softmax(dim=1)

    def forward(self, x: tensor) -> tensor:
        inputs = x
        conv1 = self.conv1(inputs)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)

        middle = self.middle(pool4)

        unconv1 = self.unconv1(middle)
        up4 = self.up4(cat((conv4, unconv1), dim=1))
        # print(unconv1.shape)
        # print(conv4.shape)
        # exit()
        unconv2 = self.unconv2(up4)
        up3 = self.up3(cat((conv3, unconv2), dim=1))

        unconv3 = self.unconv3(up3)
        up2 = self.up2(cat((conv2, unconv3), dim=1))

        unconv4 = self.unconv4(up2)
        up1 = self.up1(cat((conv1, unconv4), dim=1))

        post = self.post(up1)
        output = self.output(post)

        return output
