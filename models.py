import torch.nn as nn
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1, conv3x3


class ResNet_C(ResNet):
    """Resnet variant from https://arxiv.org/abs/1812.01187

    TODO
    """

    def __init__(self, block, layers, **kwargs):
        super(ResNet_C, self).__init__(block, layers, **kwargs)
        self.conv1 = nn.Sequential(
            *[
                nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
                self._norm_layer(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False),
                self._norm_layer(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
            ]
        )


class ResNet_D(ResNet):
    """Resnet variant from https://arxiv.org/abs/1812.01187

    TODO
    """

    def __init__(self, block, layers, **kwargs):
        super(ResNet_D, self).__init__(block, layers, **kwargs)
        self.conv1 = nn.Sequential(
            *[
                nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
                self._norm_layer(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False),
                self._norm_layer(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
            ]
        )

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                *[
                    nn.AvgPool2d(2, 2, ceil_mode=True, count_include_pad=False),
                    conv1x1(self.inplanes, planes * block.expansion, stride=1),
                    norm_layer(planes * block.expansion),
                ]
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)


class BasicBlockV2(nn.Module):
    r"""BasicBlock V2 from
    `"Identity Mappings in Deep Residual Networks"<https://arxiv.org/abs/1603.05027>`_ paper.
    This is used for ResNet V2 for 18 layers.
    
    Code taken from: https://github.com/pytorch/vision/pull/491
    
    Args:
        inplanes (int): number of input channels.
        planes (int): number of output channels.
        stride (int): stride size.
        downsample (Module) optional downsample module to downsample the input.
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockV2, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)  # just to make better graph
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        out = self.bn1(x)
        out = self.relu1(out)
        residual = self.downsample(out) if self.downsample is not None else x
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        return out + residual


class BottleneckV2(nn.Module):
    r"""Bottleneck V2 from
    `"Identity Mappings in Deep Residual Networks"<https://arxiv.org/abs/1603.05027>`_ paper.
    This is used for ResNet V2 for 50 layers.
    
    Code taken from: https://github.com/pytorch/vision/pull/491
    
    Args:
        inplanes (int): number of input channels.
        planes (int): number of output channels.
        stride (int): stride size.
        downsample (Module) optional downsample module to downsample the input.
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckV2, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.downsample = downsample

    def forward(self, x):

        out = self.bn1(x)
        out = self.relu1(out)

        residual = self.downsample(out) if self.downsample is not None else x
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv3(out)

        return out + residual


class ResNetV2_C(nn.Module):
    r"""ResNet V2 model from
    `"Identity Mappings in Deep Residual Networks"<https://arxiv.org/abs/1603.05027>`_ paper.
    
    Code taken from: https://github.com/pytorch/vision/pull/491
    
    Args:
        block (Module) : class for the residual block. Options are BasicBlockV1, BottleneckV1.
        layers (list of int) : numbers of layers in each block
        num_classes (int) :, default 1000, number of classification classes.
    """

    def __init__(self, block, layers, num_classes=1000):
        super(ResNetV2_C, self).__init__()
        assert block in (
            BottleneckV2,
            BasicBlockV2,
        ), "Argument block should be BottleneckV2 or BasicBlockV2"
        self.inplanes = 64

        self.conv1 = nn.Sequential(
            *[
                nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
            ]
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.bn5 = nn.BatchNorm2d(self.inplanes)
        self.relu5 = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = conv1x1(self.inplanes, planes * block.expansion, stride=stride)

        layers = [
            block(self.inplanes, planes, stride, downsample),
        ]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn5(x)
        x = self.relu5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def _resnet(arch, block, layers, variant=None, **kwargs):
    if variant is None:
        model = ResNet(block, layers, **kwargs)
    elif variant == "C":
        model = ResNet_C(block, layers, **kwargs)
    elif variant == "D":
        model = ResNet_D(block, layers, **kwargs)
    elif variant == "PA":
        model = ResNetV2_C(block, layers, **kwargs)
    return model


def resnet18(variant=None, **kwargs):
    r"""ResNet-18 model

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if variant == "PA":
        resnet_block = BasicBlockV2
    else:
        resnet_block = BasicBlock
    return _resnet("resnet18", resnet_block, [2, 2, 2, 2], variant, **kwargs)


def resnet50(variant=None, **kwargs):
    r"""ResNet-50 model

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if variant == "PA":
        resnet_block = BottleneckV2
    else:
        resnet_block = Bottleneck
    return _resnet("resnet50", resnet_block, [3, 4, 6, 3], variant, **kwargs)
