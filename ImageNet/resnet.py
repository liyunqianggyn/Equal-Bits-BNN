import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from torch.autograd import Variable
from args import args as parser_args
from utils import sign, half, halfmask

__all__ = ['resnet18_1w1a', 'resnet34_1w1a']


# custom weight binarization layer
class EBConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(EBConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        # rescaling setting, with bn layer it can be removed
        self.rescale = self.weight.clone().detach()
        fan = nn.init._calculate_correct_fan(self.weight, "fan_in")
        if parser_args.scale_fan:
            fan = fan * (1 - parser_args.prune_rate)
        gain = nn.init.calculate_gain(parser_args.nonlinearity)
        std = gain / math.sqrt(fan)
        self.rescale.data = torch.ones_like(self.weight.data) * std
        if parser_args.cuda:
            self.rescale = self.rescale.cuda()

    @property
    def clamped_scores(self):
        return self.weight.abs()

    def forward(self, x):
        # optionally EB binary codes
        if parser_args.binarization_mode == 'sign':
            subnet = sign.apply(self.weight)
            w = self.rescale * subnet

        if parser_args.binarization_mode == 'half':
            subnet = half.apply(self.weight)
            w = self.rescale * subnet

        if parser_args.binarization_mode == 'halfmask':
            subnet = halfmask.apply(self.weight, parser_args.prune_rate)
            w = self.rescale * subnet

        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


# custom activation binarization layer
class BinActive(torch.autograd.Function):
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        input = input.sign()
        return input

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input


def _weights_init(m):
    classname = m.__class__.__name__
    print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight)


class BasicBlock_1w1a(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, binary=False):
        super(BasicBlock_1w1a, self).__init__()
        self.conv1 = EBConv2d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.nonlinear = nn.PReLU(planes)
        self.conv2 = EBConv2d(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # bi-real net structure
        residual = x
        out = BinActive.apply(x)
        out = self.conv1(out)
        out = self.nonlinear(out)
        out = self.bn1(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
       
        x1 = out
        out = BinActive()(out)
        out = self.conv2(out)
        out = self.nonlinear(out)
        out = self.bn2(out)
        out += x1

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.prelu = nn.PReLU(self.inplanes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            shortcut_conv = nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, \
                                 stride=1, bias=False)
            downsample = nn.Sequential(
                shortcut_conv,
                nn.BatchNorm2d(planes * block.expansion),
                nn.AvgPool2d(kernel_size=2, stride=2)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, binary=False))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, binary=False))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)        
        x = self.prelu(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18_1w1a():
    return ResNet(BasicBlock_1w1a,  [2, 2, 2, 2])


def resnet34_1w1a():
    return ResNet(BasicBlock_1w1a,  [3, 4, 6, 3])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
