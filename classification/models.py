import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


'''
model: Resnet18
'''
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.residual_branch = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      bias=False), nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,
                      out_channels * BasicBlock.expansion,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion))

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels * BasicBlock.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion))

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_branch(x) +
                                     self.shortcut(x))


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleNeck, self).__init__()
        self.residual_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,
                      out_channels,
                      stride=stride,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,
                      out_channels * BottleNeck.expansion,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels * BottleNeck.expansion,
                          stride=stride,
                          kernel_size=1,
                          bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion))

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_branch(x) +
                                     self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, inter_layer=False):
        super(ResNet, self).__init__()
        self.inter_layer = inter_layer
        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.stage2 = self._make_layer(block, 64, layers[0], 1)
        self.stage3 = self._make_layer(block, 128, layers[1], 2)
        self.stage4 = self._make_layer(block, 256, layers[2], 2)
        self.stage5 = self._make_layer(block, 512, layers[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the 
        same as a neuron netowork layer, ex. conv layer), one layer may 
        contain more than one residual block 
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        
        Return:
            return a resnet layer
        """

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        if self.inter_layer:
            x1 = self.stage2(x)
            x2 = self.stage3(x1)
            x3 = self.stage4(x2)
            x4 = self.stage5(x3)
            x = self.avg_pool(x4)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            return [x1, x2, x3, x4, x]
        else:
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.stage4(x)
            x = self.stage5(x)
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            return x


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    # only load state_dict()
    if pretrained:
        model.load_state_dict(
            torch.load(model_urls[arch], map_location=torch.device('cpu')))

    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet50', BottleNeck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet101', BottleNeck, [3, 4, 23, 3], pretrained,
                   progress, **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet152', BottleNeck, [3, 8, 36, 3], pretrained,
                   progress, **kwargs)





'''
model: Encoder
'''
class EnDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.Encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2), # 32x32 => 16x16
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 2*32, kernel_size=3, padding=1, stride=2), # 16x16 => 8x8
            nn.GELU(),
            nn.Conv2d(2*32, 2*32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(2*32, 2*32, kernel_size=3, padding=1, stride=2), # 8x8 => 4x4
            nn.GELU(),
            nn.Flatten(), # Image grid to single feature vector
            nn.Linear(2*16*32, 384) # 特征向量压缩到384维
        )
        
        self.linear = nn.Sequential(
            nn.Linear(384, 2*16*32),
            nn.GELU()
        )

        self.Decoder = nn.Sequential(
            nn.ConvTranspose2d(2*32, 2*32, kernel_size=3, output_padding=1, padding=1, stride=2), # 4x4 => 8x8
            nn.GELU(),
            nn.Conv2d(2*32, 2*32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(2*32, 32, kernel_size=3, output_padding=1, padding=1, stride=2), # 8x8 => 16x16
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, output_padding=1, padding=1, stride=2), # 16x16 => 32x32
            nn.Tanh() # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        x = self.Encoder(x)
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.Decoder(x)
        return x