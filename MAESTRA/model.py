from __future__ import unicode_literals
import dataset
import feature
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import torch.optim as optim
from torchsummary import summary
import time
import pandas as pd
import pickle
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt


class MAESTRA(nn.Module):
    def __init__(self):
        super(MAESTRA, self).__init__()

        # 네트워크에서 반복적으로 사용되는 Conv + BatchNorm + ReLu를 합쳐서 하나의 함수로 정의
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)  # *으로 list unpacking

            return cbr

        # Contracting path
        # 1x336x431 => 64x336x431
        self.enc1_1 = CBR2d(in_channels=1, out_channels=64)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)
        # => 64x168x215
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # => 128x168x215
        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)
        # => 128x84x107
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # => 256x84x107
        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)
        # => 256x42x53
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        # => 512x42x53
        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)
        # => 512x21x26
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        # Bottleneck 구간
        # => 1024x21x26
        self.bottleNeck1 = CBR2d(in_channels=512, out_channels=1024)
        self.bottleNeck2 = CBR2d(in_channels=1024, out_channels=1024)

        # Expansive path
        # => 512x42x52
        self.unpool4 = nn.ConvTranspose2d(in_channels=1024, out_channels=512,
                                          kernel_size=2, stride=2)
        # => 512x42x52
        self.dec4_2 = CBR2d(in_channels=1024, out_channels=512)
        self.dec4_1 = CBR2d(in_channels=512, out_channels=512)
        # => 256x84x104
        self.unpool3 = nn.ConvTranspose2d(in_channels=512, out_channels=256,
                                          kernel_size=2, stride=2)
        # => 256x84x104
        self.dec3_2 = CBR2d(in_channels=512, out_channels=256)
        self.dec3_1 = CBR2d(in_channels=256, out_channels=256)
        # => 128x168x208
        self.unpool2 = nn.ConvTranspose2d(in_channels=256, out_channels=128,
                                          kernel_size=2, stride=2)
        # => 128x168x208
        self.dec2_2 = CBR2d(in_channels=256, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=128)
        # => 64X336X416
        self.unpool1 = nn.ConvTranspose2d(in_channels=128, out_channels=64,
                                          kernel_size=2, stride=2)
        # => 64X336X416
        self.dec1_2 = CBR2d(in_channels=128, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)
        # => 1X336X416
        self.fc = nn.Conv2d(64, 1, kernel_size=1, stride=1)

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=0),  # 148x244->146x242
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # ->73x121

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=0),  # ->71x119
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # ->35x59

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0),  # ->33x57
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # ->16x28

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0),  # ->14x26
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # ->7x13

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=0),  # ->5x11
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # ->2x5
        )

        self.FCN = nn.Sequential(
            nn.Linear(in_features=256 * 2 * 5, out_features=1),  # 2560->1
            nn.Sigmoid()
        )

    # __init__ 함수에서 선언한 layer들 연결해서 data propa flow 만들기
    def forward(self, x):
        layer1 = self.enc1_1(x)
        layer1 = self.enc1_2(layer1)

        out = self.pool1(layer1)

        layer2 = self.enc2_1(out)
        layer2 = self.enc2_2(layer2)

        out = self.pool2(layer2)

        layer3 = self.enc3_1(out)
        layer3 = self.enc3_2(layer3)

        out = self.pool3(layer3)

        layer4 = self.enc4_1(out)
        layer4 = self.enc4_2(layer4)

        out = self.pool4(layer4)

        bottle_neck = self.bottleNeck1(out)
        bottle_neck = self.bottleNeck2(bottle_neck)

        unpool4 = self.unpool4(bottle_neck)
        cat4 = torch.cat((transforms.CenterCrop((unpool4.shape[2], unpool4.shape[3]))
                          (layer4), unpool4), dim=1)

        dec_layer4 = self.dec4_2(cat4)
        dec_layer4 = self.dec4_1(dec_layer4)

        unpool3 = self.unpool3(dec_layer4)
        cat3 = torch.cat((transforms.CenterCrop((unpool3.shape[2], unpool3.shape[3]))
                          (layer3), unpool3), dim=1)

        dec_layer3 = self.dec3_2(cat3)
        dec_layer3 = self.dec3_1(dec_layer3)

        unpool2 = self.unpool2(dec_layer3)
        cat2 = torch.cat((transforms.CenterCrop((unpool2.shape[2], unpool2.shape[3]))
                          (layer2), unpool2), dim=1)

        dec_layer2 = self.dec2_2(cat2)
        dec_layer2 = self.dec2_1(dec_layer2)

        unpool1 = self.unpool1(dec_layer2)
        cat1 = torch.cat((transforms.CenterCrop((unpool1.shape[2], unpool1.shape[3]))
                          (layer1), unpool1), dim=1)
        # cat1 = torch.cat((unpool1, layer1), dim=1)

        dec_layer1 = self.dec1_2(cat1)
        dec_layer1 = self.dec1_1(dec_layer1)

        out = self.fc(dec_layer1)
        out = self.features(out)
        out = torch.flatten(out, 1)
        out = self.FCN(out)

        return out