import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pickle
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
    #"vggface2": "https://onedrive.live.com/download?cid=D07B627FBE5CFFFA&resid=D07B627FBE5CFFFA%21587&authkey=APXT_JMvytW7cgk",
    "vggface2": "/kaggle/input/resnet50-vggface2-weight/resnet50_scratch_weight.pkl",
    "vggface2_ft": "/kaggle/input/resnet50-vggface2-weight/resnet50_ft_weight.pkl",
}

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

#### CBAM BLOCK
import os
import torch
import torch.nn as nn
from .resnet_cbam_pytorchcv.common import conv1x1_block, conv7x7_block, calc_net_weights
from .resnet_cbam_pytorchcv.resnet import ResInitBlock, ResBlock, ResBottleneck


class MLP(nn.Module):
    """
    Multilayer perceptron block.

    Parameters
    ----------
    channels : int
        Number of input/output channels.
    reduction_ratio : int, default 16
        Channel reduction ratio.
    """
    def __init__(self,
                 channels: int,
                 reduction_ratio: int = 16):
        super(MLP, self).__init__()
        mid_channels = channels // reduction_ratio

        self.fc1 = nn.Linear(
            in_features=channels,
            out_features=mid_channels)
        self.activ = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(
            in_features=mid_channels,
            out_features=channels)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.activ(x)
        x = self.fc2(x)
        return x


class ChannelGate(nn.Module):
    """
    CBAM channel gate block.

    Parameters
    ----------
    channels : int
        Number of input/output channels.
    reduction_ratio : int, default 16
        Channel reduction ratio.
    """
    def __init__(self,
                 channels: int,
                 reduction_ratio: int = 16):
        super(ChannelGate, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.mlp = MLP(
            channels=channels,
            reduction_ratio=reduction_ratio)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        att1 = self.avg_pool(x)
        att1 = self.mlp(att1)
        att2 = self.max_pool(x)
        att2 = self.mlp(att2)
        att = att1 + att2
        att = self.sigmoid(att)
        att = att.unsqueeze(2).unsqueeze(3).expand_as(x)
        x = x * att
        return x


class SpatialGate(nn.Module):
    """
    CBAM spatial gate block.
    """
    def __init__(self):
        super(SpatialGate, self).__init__()
        self.conv = conv7x7_block(
            in_channels=2,
            out_channels=1,
            activation=None)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        att1 = x.max(dim=1)[0].unsqueeze(1)
        att2 = x.mean(dim=1).unsqueeze(1)
        att = torch.cat((att1, att2), dim=1)
        att = self.conv(att)
        att = self.sigmoid(att)
        x = x * att
        return x


class CbamBlock(nn.Module):
    """
    CBAM attention block for CBAM-ResNet.

    Parameters
    ----------
    channels : int
        Number of input/output channels.
    reduction_ratio : int, default 16
        Channel reduction ratio.
    """
    def __init__(self,
                 channels: int,
                 reduction_ratio: int = 16):
        super(CbamBlock, self).__init__()
        self.ch_gate = ChannelGate(
            channels=channels,
            reduction_ratio=reduction_ratio)
        self.sp_gate = SpatialGate()

    def forward(self, x):
        x = self.ch_gate(x)
        x = self.sp_gate(x)
        return x


class CbamResUnit(nn.Module):
    """
    CBAM-ResNet unit.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple(int, int)
        Strides of the convolution.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int | tuple[int, int],
                 bottleneck: bool):
        super(CbamResUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        if bottleneck:
            self.body = ResBottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                conv1_stride=False)
        else:
            self.body = ResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride)
        if self.resize_identity:
            self.identity_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                activation=None)
        self.cbam = CbamBlock(channels=out_channels)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = self.cbam(x)
        x = x + identity
        x = self.activ(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam = False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.use_cbam = use_cbam
        if self.use_cbam == True:
            self.CbamBlock = CbamBlock(channels = planes * 4)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.use_cbam == True:
            out = self.CbamBlock(out)
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, include_top=True, use_cbam = False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.include_top = include_top

        self.use_cbam = use_cbam
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_cbam = self.use_cbam ))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_cbam = self.use_cbam ))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        
        if not self.include_top:
            return x
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def _resnet(arch, block, layers, pretrained, progress, use_cbam = False, **kwargs):
    model = ResNet(block, layers, use_cbam = use_cbam, **kwargs)
    if pretrained == True:
        print(f'load weight in {model_urls[arch]}')
        if 'vggface2' in arch:
            with open(model_urls[arch], 'rb') as f:
                state_dict = pickle.load(f)
                
            for key in state_dict.keys():
                state_dict[key] = torch.from_numpy(state_dict[key])
        else:
            state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = _resnet(
        "resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress,num_classes = 1000, **kwargs
    )

    # model.fc = nn.Linear(512, kwargs['num_classes'])
    model.fc = nn.Linear(512, 7)
    return model


def resnet34(pretrained=True, progress=True, out_classes = 7, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = _resnet(
        "resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress,num_classes = 1000 ,**kwargs
    )
    model.fc = nn.Linear(512, out_classes)
    return model


def resnet50(pretrained=True, progress=True, out_classes = 7, use_cbam = False, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = _resnet(
        "resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress,num_classes = 1000, use_cbam = use_cbam, **kwargs
    )
    model.fc = nn.Linear(2048, out_classes)
    return model

def resnet50_vggface2(pretrained=True, progress=True,out_classes = 7,  use_cbam = False,  **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = _resnet(
        "vggface2", Bottleneck, [3, 4, 6, 3], pretrained, progress,num_classes=8631, use_cbam = use_cbam, **kwargs
    )
    model.fc = nn.Linear(2048, out_classes)
    print('model resnet50 with vggface2(trained from cratch) is done!')
    return model

def resnet50_vggface2_ft(pretrained=True, progress=True,out_classes = 7,  use_cbam = False, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = _resnet(
        "vggface2_ft", Bottleneck, [3, 4, 6, 3], pretrained, progress,num_classes=8631, use_cbam = use_cbam, **kwargs
    )
    model.fc = nn.Linear(2048, out_classes)
    print('model resnet50 with pre-train on vggface2(trained on MS1M, and then fine-tuned on VGGFace2) is done!')
    return model