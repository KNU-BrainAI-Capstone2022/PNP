from __future__ import unicode_literals
import youtube_dl
import ffmpeg
import sound_processing as sp
import pandas as pd
import os
from sklearn.preprocessing import MultiLabelBinarizer


def create_dataset(recreate=False, audio_len=10):
    if recreate:
        if os.path.exists('./MAESTRA_dataset'):
            for file in os.scandir('./MAESTRA_dataset'):
                os.remove(file.path)
            print('Removed All Dataset File')
        else:
            print('Dataset Directory Not Found')
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
        if str(row["url"]) == 'nan':
            break
        if os.path.isfile('./MAESTRA_dataset/' + str(row["id"]) + '-00.wav'):
            continue
        url, id, start, end = str(row["url"]), str(row["id"]), int(row["start"]), int(row["end"])
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            stream = ffmpeg.input('./output.m4a')
            stream = ffmpeg.output(stream, './output.wav')
        sp.split_audio('./output.wav', './MAESTRA_dataset', id, audio_len, start, end)

    print('Dataset Creation Complete')
    # ERROR: unable to download video data: HTTP Error 403: Forbidden 에러가 떴을 시
    # youtube-dl --rm-cache-dir 입력해서 해결


def create_dataframe(audio_len=10):
    metadata = pd.read_excel('./metadata.xlsx', sheet_name=0)
    data_lst = []
    mlb = MultiLabelBinarizer()

    for index, row in metadata.iterrows():
        if str(row["url"]) == 'nan':
            break
        ensemble, id, start, end = str(row["ensemble"]), str(row["id"]), int(row["start"]), int(row["end"])
        n = int((end - start) / audio_len)
        for num in range(0, n):
            audio_file = id + '-' + str(num).rjust(2, '0') + '.wav'
            labels = ensemble.split(' ')
            data_lst.append([audio_file, labels])

    data_df = pd.DataFrame(data_lst, columns=['audio_file', 'labels'])

    labels = mlb.fit_transform(data_df['labels'].values)
    new_data_df = pd.DataFrame(columns=mlb.classes_, data=labels)
    new_data_df.insert(0, 'audio_file', data_df['audio_file'])

    new_data_df.to_csv('./MAESTRA_data.csv')
    new_data_df.to_pickle('./MAESTRA_data.pkl')
    print(len(new_data_df), "개의 파일들의 데이터프레임이 완성되었습니다.")

    return new_data_df

