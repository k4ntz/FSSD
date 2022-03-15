'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from activations.torch import ReLU, Rational, TentActivation, OneSin, RARE


def make_act(act_name, learnable_k=False):
    if act_name.lower() == "relu":
        return ReLU()
    elif act_name.lower() == "rat":
        return Rational()
    elif act_name.lower() == "tent":
        return TentActivation(learnable=learnable_k)
    elif act_name.lower() == "onesin":
        return OneSin()
    elif act_name.lower() == "rare":
        return RARE("tent", k_trainable=learnable_k)
    elif act_name.lower() == "orare":
        return RARE("onesin", k_trainable=learnable_k)
    else:
        print("Unknown function")
        exit(1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, act_name=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act1 = make_act(act_name)
        self.act2 = make_act(act_name)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, act_name=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.act1 = make_act(act_name)
        self.act2 = make_act(act_name)
        self.act3 = make_act(act_name)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.act3(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, act_name=None):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, act_name=act_name)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, act_name=act_name)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, act_name=act_name)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, act_name=act_name)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.act1 = make_act(act_name)

    def _make_layer(self, block, planes, num_blocks, stride, act_name):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, act_name=act_name))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(act_name):
    return ResNet(BasicBlock, [2,2,2,2], act_name=act_name)

def ResNet34(act_name):
    return ResNet(BasicBlock, [3,4,6,3], act_name=act_name)

def ResNet50(act_name):
    return ResNet(Bottleneck, [3,4,6,3], act_name=act_name)

def ResNet101(act_name):
    return ResNet(Bottleneck, [3,4,23,3], act_name=act_name)

def ResNet152(act_name):
    return ResNet(Bottleneck, [3,8,36,3], act_name=act_name)


# def test():
#     net = ResNet18()
#     y = net(torch.randn(1,3,32,32))
#     print(y.size())

# test()
