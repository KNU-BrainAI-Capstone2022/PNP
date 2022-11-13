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
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            # layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)  # *으로 list unpacking

            return cbr

        # Contracting path
        # 1x168x216 => 64x168x216
        self.down1 = nn.Sequential(
            CBR2d(1, 64),
            CBR2d(64, 64)
        )

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.down2 = nn.Sequential(
            CBR2d(64, 128),
            CBR2d(128, 128)
        )

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.down3 = nn.Sequential(
            CBR2d(128, 256),
            CBR2d(256, 256)
        )

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.down4 = nn.Sequential(
            CBR2d(256, 512),
            CBR2d(512, 512),
            # nn.Dropout(p=0.5)
        )

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.bottleNeck = nn.Sequential(
            CBR2d(512, 1024),
            CBR2d(1024, 1024)
        )

        self.unpool4 = nn.ConvTranspose2d(in_channels=1024, out_channels=512,
                                          kernel_size=2, stride=2)

        self.up4 = nn.Sequential(
            CBR2d(1024, 512),
            CBR2d(512, 512)
        )

        self.unpool3 = nn.ConvTranspose2d(in_channels=512, out_channels=256,
                                          kernel_size=2, stride=2)

        self.up3 = nn.Sequential(
            CBR2d(512, 256),
            CBR2d(256, 256)
        )

        self.unpool2 = nn.ConvTranspose2d(in_channels=256, out_channels=128,
                                          kernel_size=2, stride=2)

        self.up2 = nn.Sequential(
            CBR2d(256, 128),
            CBR2d(128, 128)
        )

        self.unpool1 = nn.ConvTranspose2d(in_channels=128, out_channels=64,
                                          kernel_size=2, stride=2)

        self.up1 = nn.Sequential(
            CBR2d(128, 64),
            CBR2d(64, 64)
        )

        self.fc = nn.Conv2d(64, 1, kernel_size=1, stride=1)

        self.dropout = nn.Dropout(p=0.5)

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=0),  # 160x208->158x206
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # ->79x103

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=0),  # ->77x101
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # ->38x50

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0),  # ->36x48
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # ->18x24

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0),  # ->16x22
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # ->8x11

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=0),  # ->6x9
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # ->3x4
        )

        self.fc_new = nn.Sequential(
            nn.Linear(in_features=3072, out_features=1),
            nn.Sigmoid()
        )

        self.fc_new2 = nn.Sequential(
            nn.Linear(in_features=33280, out_features=1),
            nn.Sigmoid()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=256 * 3 * 4, out_features=256),  # 3072->256
            nn.ReLU(),
            # nn.BatchNorm1d(256)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=256, out_features=128),  # 256->128
            nn.ReLU(),
            # nn.BatchNorm1d(128)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(128, 1),  # 128->1
            nn.Sigmoid()
        )

        self.fc1_2 = nn.Sequential(
            nn.Linear(in_features=1 * 160 * 208, out_features=2048),  # 33280->2048
            nn.ReLU(),
            # nn.BatchNorm1d(2048)
        )

        self.fc2_2 = nn.Sequential(
            nn.Linear(in_features=2048, out_features=128),  # 2048->128
            nn.ReLU(),
            # nn.BatchNorm1d(128)
        )

        self.fc3_2 = nn.Sequential(
            nn.Linear(128, 1),  # 128->1
            nn.Sigmoid()
        )

    # __init__ 함수에서 선언한 layer들 연결해서 data propa flow 만들기
    def forward(self, x):

        layer1 = self.down1(x)

        out = self.pool1(layer1)

        layer2 = self.down2(out)

        out = self.pool2(layer2)

        layer3 = self.down3(out)

        out = self.pool3(layer3)

        layer4 = self.down4(out)

        out = self.pool4(layer4)

        bottle_neck = self.bottleNeck(out)

        unpool4 = self.unpool4(bottle_neck)
        cat4 = torch.cat((transforms.CenterCrop((unpool4.shape[2], unpool4.shape[3]))
                          (layer4), unpool4), dim=1)

        dec_layer4 = self.up4(cat4)

        unpool3 = self.unpool3(dec_layer4)
        cat3 = torch.cat((transforms.CenterCrop((unpool3.shape[2], unpool3.shape[3]))
                          (layer3), unpool3), dim=1)

        dec_layer3 = self.up3(cat3)

        unpool2 = self.unpool2(dec_layer3)
        cat2 = torch.cat((transforms.CenterCrop((unpool2.shape[2], unpool2.shape[3]))
                          (layer2), unpool2), dim=1)

        dec_layer2 = self.up2(cat2)

        unpool1 = self.unpool1(dec_layer2)
        cat1 = torch.cat((transforms.CenterCrop((unpool1.shape[2], unpool1.shape[3]))
                          (layer1), unpool1), dim=1)
        # cat1 = torch.cat((unpool1, layer1), dim=1)

        dec_layer1 = self.up1(cat1)

        out = self.fc(dec_layer1)
        out = self.features(out)
        out = torch.flatten(out, 1)
        out = self.fc_new(out)
        # out = self.fc1(out)
        # out = self.dropout(out)
        # out = self.fc2(out)
        # out = self.dropout(out)
        # out = self.fc3(out)

        return out


def conv_block_1(in_dim,out_dim, activation,stride=1):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim, kernel_size=1, stride=stride),
        nn.BatchNorm2d(out_dim),
        activation
    )
    return model


def conv_block_3(in_dim,out_dim, activation, stride=1):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(out_dim),
        activation
    )
    return model


class BottleNeck(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, activation, down=False):
        super(BottleNeck, self).__init__()
        self.down = down

        # 특성지도의 크기가 감소하는 경우
        if self.down:
            self.layer = nn.Sequential(
                conv_block_1(in_dim, mid_dim, activation, stride=2),
                conv_block_3(mid_dim, mid_dim, activation, stride=1),
                conv_block_1(mid_dim, out_dim, activation, stride=1)
            )

            # 특성지도 크기 + 채널을 맞춰주는 부분
            self.downsample = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=2)

        # 특성지도의 크기가 그대로인 경우
        else:
            self.layer = nn.Sequential(
                conv_block_1(in_dim, mid_dim, activation, stride=1),
                conv_block_3(mid_dim, mid_dim, activation, stride=1),
                conv_block_1(mid_dim, out_dim, activation, stride=1)
            )

        # 채널을 맞춰주는 부분
        self.dim_equalizer = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(self, x):
        if self.down:
            downsample = self.downsample(x)
            out = self.layer(x)
            out = out + downsample
        else:
            out = self.layer(x)
            if x.size() is not out.size():
                x = self.dim_equalizer(x)
            out = out + x
        return out


# 50-layer
class ResNet(nn.Module):

    def __init__(self, base_dim, num_classes=1, batch_size=5):
        super(ResNet, self).__init__()
        self.batch_size = batch_size
        self.activation = nn.ReLU()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(1, base_dim, 7, 2, 3),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )
        self.layer_2 = nn.Sequential(
            BottleNeck(base_dim, base_dim, base_dim * 4, self.activation),
            BottleNeck(base_dim * 4, base_dim, base_dim * 4, self.activation),
            BottleNeck(base_dim * 4, base_dim, base_dim * 4, self.activation, down=True)
        )
        self.layer_3 = nn.Sequential(
            BottleNeck(base_dim * 4, base_dim * 2, base_dim * 8, self.activation),
            BottleNeck(base_dim * 8, base_dim * 2, base_dim * 8, self.activation),
            BottleNeck(base_dim * 8, base_dim * 2, base_dim * 8, self.activation),
            BottleNeck(base_dim * 8, base_dim * 2, base_dim * 8, self.activation, down=True)
        )
        self.layer_4 = nn.Sequential(
            BottleNeck(base_dim * 8, base_dim * 4, base_dim * 16, self.activation),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.activation),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.activation),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.activation),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.activation),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.activation, down=True)
        )
        self.layer_5 = nn.Sequential(
            BottleNeck(base_dim * 16, base_dim * 8, base_dim * 32, self.activation),
            BottleNeck(base_dim * 32, base_dim * 8, base_dim * 32, self.activation),
            BottleNeck(base_dim * 32, base_dim * 8, base_dim * 32, self.activation)
        )
        self.avgpool = nn.AvgPool2d(1, 1)
        self.fc_layer = nn.Sequential(
            nn.Linear(base_dim * 32 * 6 * 7, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = self.avgpool(out)
        # out = torch.flatten(out, 1)
        out = out.view(out.shape[0], -1)
        out = self.fc_layer(out)

        return out
