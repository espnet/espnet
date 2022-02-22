import math
import numpy as np
import os
import random
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class Compose(object):
    """Compose several preprocess together.

    Args:
        preprocess (list of ``Preprocess`` objects): list of preprocess to compose.

    """

    def __init__(self, preprocess):
        self.preprocess = preprocess

    def __call__(self, sample):
        for t in self.preprocess:
            sample = t(sample)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.preprocess:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


def get_preprocessing_pipelines():
    # -- preprocess for the video stream
    preprocessing = {}
    # -- LRW config
    crop_size = (88, 88)
    preprocessing["pretrain"] = Compose([RandomCrop(crop_size), HorizontalFlip(0.5)])
    preprocessing["Train"] = preprocessing["pretrain"]

    preprocessing["Test"] = Compose([CenterCrop(crop_size)])

    preprocessing["Val"] = preprocessing["Test"]

    return preprocessing


def pad_packed_collate(batch):
    if len(batch) == 1:
        data, lengths, namelist, = zip(
            *[
                (a, a.shape[0], b)
                for (a, b) in sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
            ]
        )
        data = torch.FloatTensor(data)
        lengths = [data.size(1)]

    if len(batch) > 1:
        data_list, lengths, namelist = zip(
            *[
                (a, a.shape[0], b)
                for (a, b) in sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
            ]
        )

        if data_list[0].ndim == 3:
            max_len, h, w = data_list[
                0
            ].shape  # since it is sorted, the longest video is the first one
            data_np = np.zeros((len(data_list), max_len, h, w))
        elif data_list[0].ndim == 1:
            max_len = data_list[0].shape[0]
            data_np = np.zeros((len(data_list), max_len))
        for idx in range(len(data_np)):
            data_np[idx][: data_list[idx].shape[0]] = data_list[idx]
        data = torch.FloatTensor(data_np)
    return data, lengths, namelist


class LoadInput(Dataset):
    def __init__(self, root_path, filelist, preprocessing_func=None):

        self.datalist = self._get_paths(root_path, filelist)
        self.preprocessing_func = preprocessing_func

    def _get_paths(self, root_path, filelist):
        """Return absolute paths to all utterances, transcriptions and phoneme
        labels in the required subset."""
        datalist = []
        for speaker_id in filelist:

            datalist.append(os.path.join(root_path, speaker_id))

        return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        data_path = self.datalist[index]
        returnname = data_path.split("/")[-1]
        input_signal = torch.load(data_path)
        input_signal = self.preprocessing_func(input_signal)

        return input_signal, returnname


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

    def forward(self, x):
        x = x.unsqueeze(1)
        B, C, T, H, W = x.size()
        x = self.frontend3D(x)
        Tnew = x.shape[2]
        x = x.transpose(1, 2)
        x = x.contiguous()
        x = x.view(-1, 64, x.size(3), x.size(4))
        x = self.resnet34(x)
        x = x.view(B, Tnew, x.size(1))

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


class CenterCrop(object):
    """Crop the given image at the center"""

    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        """Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = int(round((w - tw)) / 2.0)
        delta_h = int(round((h - th)) / 2.0)
        frames = frames[:, delta_h : delta_h + th, delta_w : delta_w + tw]
        return frames


class RandomCrop(object):
    """Crop the given image at the center"""

    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        """Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.

        """
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = random.randint(0, w - tw)
        delta_h = random.randint(0, h - th)
        frames = frames[:, delta_h : delta_h + th, delta_w : delta_w + tw]
        return frames

    def __repr__(self):
        return self.__class__.__name__ + "(size={0})".format(self.size)


class HorizontalFlip(object):
    """Flip image horizontally."""

    def __init__(self, flip_ratio):
        self.flip_ratio = flip_ratio

    def __call__(self, frames):
        """Args:
            img (numpy.ndarray): Images to be flipped with a probability flip_ratio
        Returns:
            numpy.ndarray: Cropped image.

        """
        t, h, w = frames.shape
        if random.random() < self.flip_ratio:
            for index in range(t):
                frames[index] = np.flip(frames[index], 1)
        return frames


def main(filedir, savedir, pretrainedmodel, dset, ifcuda, debug=False):
    NWORK = 15
    filelist = os.listdir(filedir)
    if ifcuda == "true":
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    if debug == "true":
        debug = True
    else:
        debug = False

    model = Lipreading("temporalConv", inputDim=256, hiddenDim=512)
    self_state = model.state_dict()
    if ifcuda is True:
        loaded_state = torch.load(pretrainedmodel)
    else:
        loaded_state = torch.load(pretrainedmodel, map_location="cpu")
    loaded_state = {k: v for k, v in loaded_state.items() if k in self_state}
    self_state.update(loaded_state)
    model.load_state_dict(self_state)
    model = model.to(device)

    preprocessing = get_preprocessing_pipelines()
    dataset = LoadInput(filedir, filelist, preprocessing_func=preprocessing[dset])
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=pad_packed_collate,
        num_workers=NWORK,
    )
    for count, batch in enumerate(data_loader, 0):
        data = batch[0].to(device)
        datalength = list(batch[1])
        name = list(batch[2])

        try:
            with torch.no_grad():
                features = model(data)
            for i in range(len(datalength)):
                savedata = features[i, : datalength[i], :]
                torch.save(
                    savedata,
                    savedir + "/" + name[i],
                    _use_new_zipfile_serialization=False,
                )
                if debug is True:
                    print("Makefeatures for " + name[i])
        except Exception as e:
            print(e)


# hand over parameter overview
# sys.argv[1] = filedir, source directory of video frame pictures
# sys.argv[2] = savedir, savedirectory for features
# sys.argv[3] = pretrainedmodel, Path to pretrained video model
# sys.argv[4] = dset, dataset part (Train, Test, Val, pretrain)
# sys.argv[5] = ifcuda, if use cuda
# optional
# sys.argv[6] = debug, if debug should be used


if len(sys.argv) > 6:
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
else:
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
