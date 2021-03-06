import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from torch.autograd import Variable
from args import args as parser_args
from utils import sign, half, halfmask

__all__ = ['resnet18', 'resnet34']


# custom weight binarization layer
class EBConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(EBConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        # rescaling setting
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


# basic blocks: bi-real net structure
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = EBConv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.nonlinear = nn.PReLU(planes)
        self.conv2 = EBConv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride !=1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        # bi-real net structure
        out = BinActive.apply(x)
        out = self.conv1(out)
        out = self.nonlinear(out)
        out = self.bn1(out)
        out += self.shortcut(x)

        x1 = out
        out = BinActive.apply(out)
        out = self.conv2(out)
        out = self.nonlinear(out)
        out = self.bn2(out)
        out += x1
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.prelu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# generate network family
def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def test():
    net = resnet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
