import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock, conv1x1


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


class ResNet_D(ResNet_C):
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


def _resnet(arch, block, layers, variant=None, **kwargs):
    if variant is None:
        model = ResNet(block, layers, **kwargs)
    elif variant == "C":
        model = ResNet_C(block, layers, **kwargs)
    elif variant == "D":
        model = ResNet_D(block, layers, **kwargs)
    return model


def resnet18(variant=None, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], variant, **kwargs)


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing required in https://arxiv.org/abs/1812.01187

    Code taken from: https://github.com/seominseok0429/label-smoothing-visualization-pytorch
    """

    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()

    def forward(self, x, target, smoothing=0.1):
        confidence = 1.0 - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()
