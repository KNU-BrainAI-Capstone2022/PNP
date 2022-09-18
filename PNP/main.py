from __future__ import unicode_literals
import dataset_edit
import feature_extract
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchsummary import summary
import pandas as pd
import pickle
from IPython.display import display
import numpy as np
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('학습을 진행하는 기기:', device)
print('cuda index:', torch.cuda.current_device())
print('gpu 개수:', torch.cuda.device_count())
print("graphic name:", torch.cuda.get_device_name())


class MAESTRAdata(torch.utils.data.Dataset):
    def __init__(self, train=True, transform=None):
        self.train = train
        self.transform = transform
        self.file = './MAESTRA_data.pkl'
        label_lst = ['cello', 'piano']

        with open(self.file, "rb") as f:
            pkl_data = pickle.load(f)
        split_idx = pkl_data.index[pkl_data['audio_file'] == '1028-00.wav']

        if self.train:
            self.labels = np.array(pkl_data.loc[:split_idx[0] - 1, label_lst].values.tolist())
            self.audio_path = pkl_data.loc[:split_idx[0] - 1, 'audio_file'].values.tolist()
        else:
            self.labels = np.array(pkl_data.loc[split_idx[0]:, label_lst].values.tolist())
            self.audio_path = pkl_data.loc[split_idx[0]:, 'audio_file'].values.tolist()

        print(str(self.labels.shape[0]) + '개의 데이터셋 특성 추출 중..')
        dur = 1
        self.data = np.array([])
        for n in self.audio_path:
            self.data = np.append(self.data, feature_extract.make_cqt('./MAESTRA_dataset/' + n))
            if dur % 50 == 0:
                print(str(dur) + '/' + str(self.labels.shape[0]))
            dur += 1
        self.data = self.data.reshape((self.labels.shape[0], 1, 168, 216))
        print('데이터 특성 추출 완료')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]

        if self.transform is not None:
            data = self.transform(data)

        return data, label


# dataset_edit.create_dataset(recreate=False, audio_len=5)

# feature_extract.compare_features('./MAESTRA_dataset/1020-02.wav')
# feature_extract.show_cqt('./MAESTRA_dataset/1020-02.wav')
# 84x431 -> hop_length 512->1024 ->> 168x216

# train_df = dataset_edit.create_dataframe(audio_len=5)
# print(train_df)

# with open('MAESTRA_data.pkl', "rb") as file:
#     data = pickle.load(file)
# split_idx = data.index[data['audio_file'] == '1028-00.wav']
# print(split_idx)
# labels = data.loc[:split_idx[0] - 1, ['audio_file']].values.tolist()
# print(labels)
# print(len(labels))
# print(data)

# display(data.iloc[0])
# display(data.loc[:, 'cello'])

# label_lst = ['cello', 'piano']
# labels = data[label_lst].values.tolist()
# print(labels[0])

batch_size = 5
train_dataset = MAESTRAdata(train=True, transform=None)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=0)

test_dataset = MAESTRAdata(train=False, transform=None)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False, num_workers=0)

dataiter = iter(train_loader)
images, labels = dataiter.next()
print(images.size())
print(labels)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, padding=1),  # 168x216->168x216
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # ->84x108

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1),  # ->84x108
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # ->42x54

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(25, 3), padding=1),  # ->20x54
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # ->10x27
        )

        self.FCN = nn.Sequential(
            nn.Linear(in_features=32 * 10 * 27, out_features=1024),  # 8640->1024
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.FCN(x)
        return x


model = CNN().to(device)
summary(model, input_size=(1, 168, 216))

criterion = nn.BCEWithLogitsLoss()
# criterion = nn.MultiLabelSoftMarginLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
dataloader_dict = {'train': train_loader, 'val': test_loader}

plot_acc = []
train_loss = []
test_loss = []
n_labels = 2


def train_model(model, dataloader_dict, criterion, optimizer, num_epoch):
    since = time.time()
    best_acc = 0.0

    for epoch in range(num_epoch):
        print('Epoch {}/{}'.format(epoch + 1, num_epoch))
        print('-' * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            for inputs, targets in dataloader_dict[phase]:
                inputs = inputs.to(device).float()
                targets = targets.type(torch.FloatTensor)
                targets = targets.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs).to(device)
                    loss = criterion(outputs.view_as(targets), targets)
                    # _, preds = torch.max(outputs, 1)

                    # print(outputs)
                    # print(targets)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)

                    # is_correct = True
                    # for i in range(n_labels):
                    #     if (outputs[i] > 0.5).type(torch.FloatTensor) != targets[i]:
                    #         is_correct = False
                    # if is_correct:
                    #     epoch_corrects += 1
                    # epoch_corrects += torch.sum(preds == targets.data)

            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            # epoch_acc = epoch_corrects.double() / len(dataloader_dict[phase].dataset)

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val':
                test_loss.append(epoch_loss)
                # plot_acc.append(epoch_acc)
            else:
                train_loss.append(epoch_loss)
            # if phase == 'val' and epoch_acc > best_acc:
            #     best_acc = epoch_acc

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    return model


num_echos = 15
model = train_model(model, dataloader_dict, criterion, optimizer, num_echos)

torch.save(model.state_dict(), "./MAESTRAmodel.pt")

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.to(device).float()
        labels = labels.type(torch.LongTensor)
        labels = labels.to(device)
        outputs = model(images)
        print(outputs)
        print(labels.data)