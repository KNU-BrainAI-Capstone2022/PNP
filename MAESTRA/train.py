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
from time import time
import pandas as pd
import pickle
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt


def t_train(instrument, model, dataloader_dict, criterion, optimizer, num_epoch, device):
    start = time()
    best_acc = 0.0
    inst_dic = {'cello': 0, 'clarinet': 1, 'drum': 2, 'flute': 3,
                'piano': 4, 'viola': 5, 'violin': 6}
    target_lbl = inst_dic[instrument]

    print("Begin training...")
    # 훈련 루프
    for epoch in range(num_epoch):
        n = 0
        running_acc = 0.0
        epoch_loss = 0.0
        total = 0

        print('Epoch {}/{}'.format(epoch + 1, num_epoch))
        print('-' * 20)

        for inputs, targets in dataloader_dict['train']:
            inputs = inputs.to(device).float()
            targets = targets[:,target_lbl].unsqueeze(1).float().to(device)

            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            n += 1
            if n % 100 == 0:
                print(f"Try : {n:4d}, Loss : {epoch_loss/n:.3f}")
                now = time() - start
                print('훈련 진행 중.. {:.0f}m {:.0f}s'.format(now // 60, now % 60))

        epoch_loss = epoch_loss / len(dataloader_dict['train'].dataset)
        print('Train Loss: {:.4f}'.format(epoch_loss))

        # 검증 루프
        with torch.no_grad():
            model.eval()

            for inputs, targets in dataloader_dict['val']:
                inputs = inputs.to(device).float()
                targets = targets[:,target_lbl].unsqueeze(1).float().to(device)

                output = model(inputs)
                val_loss = criterion(output, targets)

                _, predicted = torch.max(output, 1)
                epoch_loss += val_loss.item()
                total += targets.size(0)
                running_acc += (predicted == targets).sum().item()

        epoch_loss = epoch_loss / len(dataloader_dict['val'])
        print('Val Loss: {:.4f}'.format(epoch_loss))

        accuracy = (100 * running_acc / total)

        if accuracy > best_acc:
            best_acc = accuracy

        print('Complted training batch', epoch, 'Training Loss is: %.4f' % epoch_loss,
              'Validation Loss is: %.4f' % epoch_loss, 'Accuracy is %d %%' % best_acc)

        print()

    time_elapsed = time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))