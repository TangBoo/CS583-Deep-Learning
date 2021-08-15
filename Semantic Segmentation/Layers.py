import torch.nn as nn
import torch
import Config as config


class ResidualConv(nn.Module):
    def __init__(self, input_channel, output_channel, padding=1, down_sample='conv'):
        super(ResidualConv, self).__init__()
        if down_sample not in ['conv', 'pool', 'none']:
            raise ValueError
        self.downSample = down_sample
        stride = 1 if down_sample == 'pool' or down_sample == 'none' else 2
        self.conv_block = nn.Sequential(
            nn.GroupNorm(num_groups=config.Num_groups, num_channels=input_channel),
            nn.ReLU(),
            nn.Conv2d(input_channel, output_channel, stride=stride, kernel_size=3, padding=padding),

            nn.GroupNorm(num_groups=config.Num_groups, num_channels=output_channel),
            nn.ReLU(),
            nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=1),
        )

        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(num_groups=config.Num_groups, num_channels=output_channel)
        )
        if stride == 1 and down_sample == 'pool':
            self.pooling = nn.MaxPool2d(stride=2, kernel_size=2)
        else:
            self.pooling = lambda x: x

    def forward(self, x):
        x = self.conv_block(x) + self.conv_skip(x)
        return self.pooling(x)


class Upsample(nn.Module):
    def __init__(self, input_channel, stride, interpolate='none'):
        super(Upsample, self).__init__()
        if interpolate not in ['none', 'bilinear', 'nearest', 'bicubic']:
            raise ValueError
        self.interpolate = interpolate
        if interpolate == 'none':
            kernel = 2 * stride - stride % 2
            self.upsample = nn.ConvTranspose2d(input_channel, input_channel, kernel_size=kernel, stride=stride,
                                               padding=1)
        elif interpolate is not 'none':
            self.upsample = nn.Upsample(scale_factor=stride, mode=interpolate)
        self.conv = nn.Conv2d(input_channel, input_channel, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x


if __name__ == '__main__':
    res_block = ResidualConv(64, 128, 1, 'pool').cuda()
    x = torch.rand((1, 64, 256, 256)).cuda()
    res = res_block(x)
