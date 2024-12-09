import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule
import copy
 
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(BasicBlock, self).__init__()
        self.norm = norm
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(self.expansion*planes, self.expansion*planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(Bottleneck, self).__init__()
        self.norm = norm
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(self.expansion*planes, self.expansion*planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(self.expansion*planes, self.expansion*planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out, out, out, out, out, out


class ResNet(BasicModule):
    def __init__(self, model_name, block, num_blocks, channel=3, num_classes=10, norm='instancenorm'):
        super(ResNet, self).__init__()
        self.model_name = model_name
        self.in_planes = 64
        self.norm = norm

        self.conv1 = nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(64, 64, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.adaptiveAvgPool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, out_feature=False):
        out1 = F.relu(self.bn1(self.conv1(x)))
        out2 = self.layer1(out1)  
        out3 = self.layer2(out2)  
        out4 = self.layer3(out3)  
        out5 = self.layer4(out4)  
        out6 = self.adaptiveAvgPool2d(out5)  #512*1*1
        feature = out6.view(out6.size(0), -1)
        out = self.linear(feature)
        if out_feature == False:
            return out
        else:
            return out,feature,out5.view(out5.size(0), -1),out4.view(out4.size(0), -1),out3.view(out3.size(0), -1),out2.view(out2.size(0), -1)
    
    def pass_remain_layers(self, x, out_feature=False):
        out3 = self.layer2(x)  
        out4 = self.layer3(out3)  
        out5 = self.layer4(out4)  
        out6 = F.avg_pool2d(out5, 4)  
        feature = out6.view(out6.size(0), -1)
        out = self.linear(feature)
        if out_feature == False:
            return out
        else:
            return out,feature,out5.view(out5.size(0), -1),out4.view(out4.size(0), -1),out3.view(out3.size(0), -1)

def ResNet18(channel=3, num_classes=10, norm='batchnorm'):
    model_name = 'resnet18'
    return ResNet(model_name,BasicBlock, [2,2,2,2], channel=channel, num_classes=num_classes, norm=norm)

def ResNet34(num_classes=10):
    model_name = 'resnet34'
    return ResNet(model_name, BasicBlock, [3,4,6,3], num_classes)
 
def ResNet50(num_classes=10):
    model_name = 'resnet50'
    return ResNet(model_name, Bottleneck, [3,4,6,3], num_classes)
 
def ResNet101(num_classes=10):
    model_name = 'resnet101'
    return ResNet(model_name, Bottleneck, [3,4,23,3], num_classes)
 
def ResNet152(num_classes=10):
    model_name = 'resnet152'
    return ResNet(model_name, Bottleneck, [3,8,36,3], num_classes)
 
if __name__ == '__main__':
    net = ResNet18(num_classes=10)
    y = net(torch.randn(20, 3, 112, 112), out_feature=True)
    print(y[0].shape,y[1].shape)
   