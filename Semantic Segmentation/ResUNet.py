import torch
from torch import nn
from Layers import ResidualConv, Upsample
import Config as config
import time


class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        if name is None:
            prefix = r"/home/aiteam_share/database/ISLES2018_manual_aif/train_3DUnet/checkpoint/" + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)


class ResUNet(BasicModule):
    def __init__(self, input_channel, output_channel, base=64, depth=4, mode='seg'):
        '''
        :param input_channel:
        :param output_channel:
        :param base:
        :param depth:
        :param mode: seg or det
        Describe:
            As for encode part, control the create of encoder using base and base_, base is always two time of base_, base as output channel, base_ as input channel.
            As for decode part, base_ is as input channel, base is as output channel, base is always two time of base_.
            The bridge skip is connection between encoder and decoder, it only expand feature maps as channel dimension
        '''
        super(ResUNet, self).__init__()
        if mode not in ['seg', 'det']:
            raise ValueError
        self.mode = mode
        self.depth = depth
        self.input_layer = nn.Sequential(
            nn.Conv2d(input_channel, base, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=config.Num_groups, num_channels=base),
            nn.ReLU(),
            nn.Conv2d(base, base, kernel_size=3, padding=1)
        )
        self.input_skip = nn.Conv2d(input_channel, base, kernel_size=3, stride=1, padding=1)
        encoder_layers = []

        # base = 32
        for ii in range(depth - 1):
            base_ = base  # 32, 64, 128, 256
            base *= 2  # 64, 128, 256
            encoder_layers.append(ResidualConv(base_, base, padding=1, down_sample='pool'))

        self.encoder_layers = nn.ModuleList(encoder_layers)
        # ------------bridge between encoder and decoder------------------
        base_ = base  # 256
        base *= 2  # 512
        self.bridge_skip = ResidualConv(base_, base, padding=1, down_sample='none')

        # ------------Decoder------------------
        decoder_layers = []
        for ii in range(depth - 1):  # 1024->512, 512->256, 256->128
            decoder_layers.append(nn.Sequential(
                Upsample(base, stride=2, interpolate='none'),
                ResidualConv(base, base_, padding=1, down_sample='none')  # 512 +128, 256 + 64, 128 + 32
            ))
            base = base_ + base_ // 2
            base_ //= 2

        self.decoder_layers = nn.ModuleList(decoder_layers)
        # base->1
        self.output = nn.Sequential(
            ResidualConv(base, base_, padding=1, down_sample='none'),
            nn.Conv2d(base_, output_channel, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x = self.input_layer(x) + self.input_skip(x)
        Xs = [x]
        for layer in self.encoder_layers:
            x = layer(x)
            Xs.append(x)

        x = self.bridge_skip(x)
        proposals = [x]
        Xs.pop()
        Xs.reverse()
        for idx, layer in enumerate(self.decoder_layers):
            proposals.append(layer(x))
            x = torch.cat((Xs[idx], proposals[-1]), 1)
        x = self.output(x)
        proposals.append(x)
        if self.mode == 'seg':
            return x
        return proposals


if __name__ == "__main__":
    model = nn.DataParallel(ResUNet(1, 1, base=64, depth=6)).cuda()
    print(model.module)
    x = torch.rand((1, 1, 256, 256)).cuda()
    # res = model(x)
    # for feature in res:
    #     print(feature.shape)
