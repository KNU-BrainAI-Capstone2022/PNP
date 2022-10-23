from __future__ import unicode_literals
import youtube_dl
import ffmpeg
import sound
import pandas as pd
import os
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import feature


# 데이터셋을 만들기 위한 파일 (d:dataset)
class maestraDataset(torch.utils.data.Dataset):
    def __init__(self, train=True, transform=None):
        """
        :param train: True->훈련 데이터셋, False->테스트 데이터셋
        :param transform: Data augmentation 커스텀
        """
        self.train = train
        self.transform = transform
        self.file = './MAESTRA_data.pkl'
        self.data = np.array([])
        # 라벨 리스트 정의
        label_lst = ['cello', 'piano']
        # 데이터 프레임 불러오기
        with open(self.file, "rb") as f:
            pkl_data = pickle.load(f)
        # train과 test를 나눌 인덱스 정의
        split_idx = pkl_data.index[pkl_data['audio_file'] == '2000-000.wav']
        # train, test 여부에 따라 파일명(ex:1000-000.wav), 라벨 목록(ex:[0,1,0,0])을 순서대로 리스트에 할당
        if self.train:
            self.labels = np.array(pkl_data.loc[:split_idx[0] - 1, label_lst].values.tolist())
            self.audio_path = pkl_data.loc[:split_idx[0] - 1, 'audio_file'].values.tolist()
        else:
            self.labels = np.array(pkl_data.loc[split_idx[0]:, label_lst].values.tolist())
            self.audio_path = pkl_data.loc[split_idx[0]:, 'audio_file'].values.tolist()
        """
        print(str(self.labels.shape[0]) + 'dataset feature being extracted..')
        dur = 1
        # 모든 오디오 데이터의 특성을 추출해서 변환 후 저장
        for path in self.audio_path:
            self.data = np.append(self.data, feature.f_cqt('./MAESTRA_dataset/' + path))
            # 50번째 마다 진행 상황을 출력
            if dur % 50 == 0:
                print(str(dur) + '/' + str(self.labels.shape[0]))
            dur += 1
        # 데이터 크기를 모델 입력 크기에 맞게 재설정
        self.data = self.data.reshape((self.labels.shape[0], 1, 168, 431))
        print('All data feature extraction complete')
        """
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 오디오 데이터의 특성을 추출해서 변환 후 저장
        self.data = np.append(self.data, feature.f_cqt('./MAESTRA_dataset/' + self.audio_path[idx]))
        data = self.data

        if self.transform is not None:
            data = self.transform(data)

        label = self.labels[idx]

        return data, label


def d_extract(re_extract=False, audio_len=10):
    """
    :param re_extract: 모든 데이터셋 초기화 후 재추출 여부
    :param audio_len: 데이터셋 하나 당 길이
    """
    if re_extract:
        # Dataset 보관 폴더 존재 여부 확인
        if os.path.exists('./MAESTRA_dataset'):
            # 폴더 내 모든 Dataset 파일 제거
            for file in os.scandir('./MAESTRA_dataset'):
                os.remove(file.path)
            print('Removed all dataset file')
        else:
            print('Cannot find dataset directory')
            return

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'output.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
    }

    metadata = pd.read_excel('./metadata.xlsx', sheet_name=0)
    print(metadata)

    for index, row in metadata.iterrows():
        # 비어있는 url 셀이 있을 때 까지 읽어들이며 Dataset을 생성
        if str(row["url"]) == 'nan':
            break
        # Dataset 중복 생성 방지
        if os.path.isfile('./MAESTRA_dataset/' + str(row["id"]) + '-000.wav'):
            continue

        url, id, start, end = str(row["url"]), str(row["id"]), int(row["start"]), int(row["end"])
        # 데이터 추출
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            stream = ffmpeg.input('./output.m4a')
            stream = ffmpeg.output(stream, './output.wav')
        # audio_len 초씩 나눈 Dataset 저장
        sound.s_split('./output.wav', './MAESTRA_dataset', id, audio_len, start, end)

    print('Dataset extraction complete')
    # ERROR: unable to download video data: HTTP Error 403: Forbidden 에러가 떴을 시
    # youtube-dl --rm-cache-dir 입력해서 해결


def d_create_df(audio_len=10):
    metadata = pd.read_excel('./metadata.xlsx', sheet_name=0)
    data_lst = []
    mlb = MultiLabelBinarizer()

    for index, row in metadata.iterrows():
        # 비어있는 url 셀이 있을 때 까지 읽어들이며 Data list를 생성
        if str(row["url"]) == 'nan':
            break

        ensemble, id, start, end = str(row["ensemble"]), str(row["id"]), int(row["start"]), int(row["end"])
        n = int((end - start) / audio_len)
        # 각 데이터셋의 [file name, label]을 리스트에 저장
        for num in range(0, n):
            audio_file = id + '-' + str(num).rjust(3, '0') + '.wav'
            labels = ensemble.split(' ')
            data_lst.append([audio_file, labels])
    # file name과 labels를 열로 하는 데이터 프레임을 생성
    data_df = pd.DataFrame(data_lst, columns=['audio_file', 'labels'])
    # 모든 label 성분을 자동으로 분류하고, 각 label을 열로 하는 새 데이터 프레임 생성
    labels = mlb.fit_transform(data_df['labels'].values)
    new_data_df = pd.DataFrame(columns=mlb.classes_, data=labels)
    new_data_df.insert(0, 'audio_file', data_df['audio_file'])
    # 새 데이터 프레임을 csv파일과 pickle파일 형태로 저장
    new_data_df.to_csv('./MAESTRA_data.csv')
    new_data_df.to_pickle('./MAESTRA_data.pkl')
    print(len(new_data_df), "개의 파일들의 데이터프레임이 완성되었습니다.")

    return new_data_df
