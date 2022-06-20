from __future__ import unicode_literals
import youtube_dl
import ffmpeg
import preprocess
import feature_extract
import os
import torch
import torch.nn as nn
import numpy as np

USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('학습을 진행하는 기기:', device)

print('cuda index:', torch.cuda.current_device())
print('gpu 개수:', torch.cuda.device_count())
print("graphic name:", torch.cuda.get_device_name())

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(1, 10), stride=(1, 2)),  # 2584->1288
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # ->644

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(1, 10), stride=(1, 2)),  # ->318
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # ->159

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 9), stride=(1, 2)),  # ->76
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # ->38

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 8), stride=(1, 2)),  # ->16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # ->8

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 3)),  # ->6
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))  # ->3
        )

        self.FCN = nn.Sequential(
            nn.Linear(in_features=128 * 30 * 3, out_features=2048),  # 11520->2048
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=21)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.FCN(x)
        return x

ydl_opts = {
    'format' : 'bestaudio/best',
    'outtmpl': 'output.%(ext)s',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
    }],
}

with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download(["https://youtu.be/gosY-UrpHcA"])
    stream = ffmpeg.input('output.m4a')
    stream = ffmpeg.output(stream, 'output.wav')

n_data = preprocess.split_music('./output.wav', './', 'output')

af_test_abspath = os.path.abspath('./')
id_ = 'output'
new_test_features = []

for i in range(n_data + 1):
    af_file_name = os.path.join(af_test_abspath + '/' + id_ + '-' + str(i).rjust(2, '0') + '.wav')
    data = feature_extract.make_mfcc(af_file_name)
    new_test_features.append([data])

new_data = np.array(new_test_features)
images = torch.Tensor(new_data.reshape(n_data+1,1,30,2584))

modelpath = './PNPmodel_1.pt'

model = CNN()
model.load_state_dict(torch.load(modelpath))

outputs = model(images)
_, predicted = torch.max(outputs.data, 1)

print(predicted)