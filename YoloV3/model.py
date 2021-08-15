import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable as V
from torchvision import transforms as T

# Tuple: convolution block [output_channel, kernel_size, stride]
# B is residual block followed by the number of repeats
# S is for a scale prediction block and computing the yolo loss
# U is for upsampling the feature map
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    # first route from the end of the previous block
    (512, 3, 2),
    ["B", 8],
    # second route from the end of the previous block
    (1024, 3, 2),
    ["B", 4],
    # until here is YOLO-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]


# This layer also allows us to toggle the bn_act to false and skip the batch normalization \
# and activation function which we will use in the last layer before output.

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    @staticmethod
    def forward(x):
        return x * torch.tanh(F.softplus(x))


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=(not bn_act), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.mish = Mish()
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            x = self.conv(x)
            x = self.bn(x)
            x = self.mish(x)
            return x
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super(ResidualBlock, self).__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1)
                )
            ]
        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + self.use_residual * x
        return x


# The last predefined block we will use is the ScalePrediction
# which is the last two convolutional layers leading up to the prediction for each scale.
class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes, anchors_per_scale=3):
        super(ScalePrediction, self).__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(2 * in_channels, (num_classes + 5) * anchors_per_scale, bn_act=False, kernel_size=1)
        )
        self.num_classes = num_classes
        self.anchors_per_scale = anchors_per_scale

    def forward(self, x):
        # x : [batch_size, channels, height, width]
        x = self.pred(x).view(x.shape[0], self.anchors_per_scale, self.num_classes + 5, x.shape[2], x.shape[3]).permute(
            0, 1, 3, 4, 2).contiguous()
        return x


class YOLOv3(nn.Module):
    def __init__(self, in_channels=1, num_classes=80):
        super(YOLOv3, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

    def forward(self, x):
        outputs = []
        route_connections = []

        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)
            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)
            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()
        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels
        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(
                    ResidualBlock(
                        in_channels,
                        num_repeats=num_repeats
                    )
                )

            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes, anchors_per_scale=3)
                    ]
                    in_channels = in_channels // 2
                elif module == "U":
                    layers.append(
                        nn.Upsample(scale_factor=2),
                    )
                    in_channels = in_channels * 3

        return layers


def test():
    num_classes = 3
    model = YOLOv3(num_classes=num_classes).cuda()
    print(model)
    img_size = 1024
    x = torch.randn((2, 1, img_size, img_size)).cuda()
    out = model(x)
    step = 32
    for o in out:
        print(o.shape, img_size // step, img_size // step, 5 + num_classes)
        step /= 2
    assert out[0].shape == (2, 3, img_size // 32, img_size // 32, 5 + num_classes)
    assert out[1].shape == (2, 3, img_size // 16, img_size // 16, 5 + num_classes)
    assert out[2].shape == (2, 3, img_size // 8, img_size // 8, 5 + num_classes)
    return


if __name__ == "__main__":
    test()
