import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchsummary import summary
import time
import numpy as np
import pandas as pd


class PNPdata(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None):
        self.train = train

        if self.train:
            self.file = root + '/train_data.json'
        else:
            self.file = root + '/test_data.json'

        self.features = pd.read_json(self.file)
        self.data = np.array(self.features.feature.tolist())
        self.label = np.array(self.features.class_label.tolist())

        if self.train:
            self.data = self.data.reshape((1860, 1, 30, 2584))
        else:
            self.data = self.data.reshape((19, 1, 30, 2584))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]

        if self.transform is not None:
            data = self.transform(data)

        return data, label


dataset = PNPdata(root='./processed_dataset', train=True, transform=None)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
test_dataset = PNPdata(root='./processed_dataset', train=False, transform=None)

batch_size = 5
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False, num_workers=0)

dataloader_dict = {'train': train_loader, 'val': val_loader}

USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('학습을 진행하는 기기:', device)

print('cuda index:', torch.cuda.current_device())
print('gpu 개수:', torch.cuda.device_count())
print("graphic name:", torch.cuda.get_device_name())

dataiter = iter(train_loader)
images, labels = dataiter.next()
print(images.size())


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


model = CNN().to(device)
summary(model, input_size=(1, 30, 2584))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
plot_acc = []
train_loss = []
val_loss = []


def train_model(model, dataloader_dict, criterion, optimizer, num_epoch):
    since = time.time()
    best_acc = 0.0

    for epoch in range(num_epoch):
        print('Epoch {}/{}'.format(epoch + 1, num_epoch))  # count epoch
        print('-' * 20)

        for phase in ['train', 'val']:  # 각 epoch마다 train과 test phase가 있음
            if phase == 'train':
                model.train()  # 모델 훈련 진행 (*dropout)
            else:
                model.eval()  # 모델 평가 진행

            epoch_loss = 0.0
            epoch_corrects = 0

            for inputs, label in dataloader_dict[phase]:  # dataloader로부터 dataset과 이에 대응하는 label을 불러옴
                inputs = inputs.to(device).float()  # GPU로 input data를 올림
                label = label.type(torch.LongTensor)
                label = label.to(device)  # GPU로 label을 올림
                optimizer.zero_grad()  # Gradient를 0으로 초기화

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)  # output layer에서 가장 값이 큰 1개의 class를 예측 값으로 set
                    loss = criterion(outputs, label)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # 통계
                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == label.data)

            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloader_dict[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val':
                val_loss.append(epoch_loss)
                plot_acc.append(epoch_acc)
            else:
                train_loss.append(epoch_loss)
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return model


print('====================Finished modeling')

# Lenet 모델 실행
num_echos = 20
model = train_model(model, dataloader_dict, criterion, optimizer, num_echos)

torch.save(model.state_dict(), "./PNPmodel.pt")

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.to(device).float()
        labels = labels.type(torch.LongTensor)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += torch.sum(predicted == labels.data)
        print(predicted)
        print(labels.data)

print('Accuracy of the network on the 19 test images: {:.4f} %'.format(100 * correct // 19))