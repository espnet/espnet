import torch
import math, os
import numpy as np
import torch
import random
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


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


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.bnfc = nn.BatchNorm1d(num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bnfc(x)
        return x


class Lipreading(nn.Module):
    def __init__(self, mode, inputDim=256, hiddenDim=512):
        super(Lipreading, self).__init__()
        self.mode = mode
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.nLayers = 2
        # frontend3D
        self.frontend3D = nn.Sequential(
            nn.Conv3d(
                1,
                64,
                kernel_size=(5, 7, 7),
                stride=(1, 2, 2),
                padding=(2, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        # resnet
        self.resnet34 = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=self.inputDim)

        self._initialize_weights()

    def forward(self, x):
        ifcuda = x.is_cuda
        x = x.unsqueeze(1)
        if self.training is True:
            x = RandomCrop(x, (88, 88))
            x = HorizontalFlip(x)
        else:
            x = CenterCrop(x, (88, 88))
        if ifcuda is True:
            x = x.cuda()
        else:
            pass
        x = self.frontend3D(x)
        x = x.transpose(1, 2)
        x = x.contiguous()
        x = x.view(-1, 64, x.size(3), x.size(4))
        x = self.resnet34(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = (
                    m.kernel_size[0]
                    * m.kernel_size[1]
                    * m.kernel_size[2]
                    * m.out_channels
                )
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def lipreading(pretrained_video_extractor, mode, inputDim=256, hiddenDim=512):
    model = Lipreading(mode, inputDim=inputDim, hiddenDim=hiddenDim)

    self_state = model.state_dict()
    loaded_state = torch.load(pretrained_video_extractor, map_location="cpu")
    loaded_state = {k: v for k, v in loaded_state.items() if k in self_state}
    self_state.update(loaded_state)
    model.load_state_dict(self_state)
    return model


def CenterCrop(batch_img, size):
    w, h = batch_img[0][0][0].shape[1], batch_img[0][0][0].shape[0]
    th, tw = size
    img = torch.zeros((len(batch_img), len(batch_img[0]), len(batch_img[0][0]), th, tw))
    for i in range(len(batch_img)):
        x1 = int(round((w - tw)) / 2.0)
        y1 = int(round((h - th)) / 2.0)
        img[i, :, :, :, :] = batch_img[i, :, :, y1 : y1 + th, x1 : x1 + tw]
    return img


def RandomCrop(batch_img, size):
    w, h = batch_img[0][0][0].shape[1], batch_img[0][0][0].shape[0]
    th, tw = size
    img = torch.zeros((len(batch_img), len(batch_img[0]), len(batch_img[0][0]), th, tw))
    for i in range(len(batch_img)):
        x1 = random.randint(0, 8)
        y1 = random.randint(0, 8)
        img[i, :, :, :, :] = batch_img[i, :, :, y1 : y1 + th, x1 : x1 + tw]
    return img


def HorizontalFlip(batch_img):
    for i in range(len(batch_img)):
        a = random.random()
        if a > 0.5:
            images = np.array(batch_img[i][0])
            for j in range(len(images)):
                images[j] = np.flip(images[j], 1)
            batch_img[i][0] = torch.tensor(images)

    return batch_img
