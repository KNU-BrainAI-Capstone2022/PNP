import pandas as pd
import os
import feature_extract as ft
import moviepy.editor as mp
from time import time
import torch

if not os.path.exists("./processed_dataset"):
    os.mkdir("./processed_dataset")
    os.mkdir("./processed_dataset/train")
    os.mkdir("./processed_dataset/test")


def split_music(input_path, output_path, audio_id):
    audio_clip = mp.AudioFileClip(input_path)
    audio_length = audio_clip.duration
    dur = 0
    num = 0
    while dur + 60 < audio_length:
        clip = audio_clip.subclip(dur, dur + 60)
        clip.write_audiofile(output_path + '\\' + audio_id + '-' + str(num).rjust(2, '0') + '.wav')
        dur += 60
        num += 1

    # clip = audio_clip.subclip(dur, int(audio_length) - 1)
    # clip.write_audiofile(output_path + '\\' + audio_id + '-' + str(num).rjust(2, '0') + '.wav')

    return num-1


if __name__ == "__main__":

    USE_CUDA = torch.cuda.is_available()
    print(USE_CUDA)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('학습을 진행하는 기기:', device)

    print('cuda index:', torch.cuda.current_device())
    print('gpu 개수:', torch.cuda.device_count())
    print("graphic name:", torch.cuda.get_device_name())

    # 각종 경로 설정
    fulldatasetpath = './musicnet'
    metadata = pd.read_csv(fulldatasetpath + '/musicnet_metadata.csv')
    bf_train_abspath = os.path.abspath(fulldatasetpath + "/train_data/")
    bf_test_abspath = os.path.abspath(fulldatasetpath + "/test_data/")
    af_train_abspath = os.path.abspath("./processed_dataset/train")
    af_test_abspath = os.path.abspath("./processed_dataset/test")

    test_id = ["1759", "1819", "2106", "2191", "2298", "2303", "2382", "2416", "2556", "2628"]
    class_map = {'Accompanied Cello': 0, 'Accompanied Clarinet': 1, 'Accompanied Violin': 2,
                 'Clarinet Quintet': 3, 'Clarinet-Cello-Piano Trio': 4, 'Horn Piano Trio': 5,
                 'Pairs Clarinet-Horn-Bassoon': 6, 'Piano Quartet': 7, 'Piano Quintet': 8,
                 'Piano Trio': 9, 'Solo Cello': 10, 'Solo Flute': 11, 'Solo Piano': 12,
                 'Solo Violin': 13, 'String Quartet': 14, 'String Sextet': 15,
                 'Viola Quintet': 16, 'Violin and Harpsichord': 17, 'Wind and Strings Octet': 18,
                 'Wind Octet': 19, 'Wind Quintet': 20}

    # 최종 데이터를 모음
    train_features = []
    test_features = []
    # 각 사운드 파일별 특징 추출
    n = 0
    start = time()
    for index, row in metadata.iterrows():
        class_label = class_map.get(row["ensemble"])
        id_ = str(row["id"])
        if id_ in test_id:
            bf_file_name = os.path.join(bf_test_abspath, id_ + ".wav")
            n_data = split_music(bf_file_name, af_test_abspath, id_)
            for i in range(n_data + 1):
                af_file_name = os.path.join(af_test_abspath + '\\' + id_ + '-' + str(i).rjust(2, '0') + '.wav')
                data = ft.make_mfcc(af_file_name)
                test_features.append([data, class_label])
        else:
            bf_file_name = os.path.join(bf_train_abspath, id_ + ".wav")
            n_data = split_music(bf_file_name, af_train_abspath, id_)
            for i in range(n_data + 1):
                af_file_name = os.path.join(af_train_abspath + '\\' + id_ + '-' + str(i).rjust(2, '0') + '.wav')
                data = ft.make_mfcc(af_file_name)
                train_features.append([data, class_label])
        n += 1
        print(n, "개 추출 완료")
    end = time()
    print('time elapsed:', end - start)

    # 리스트 -> 데이터 프레임 및 데이터 저장
    train_features_df = pd.DataFrame(train_features, columns=['feature', 'class_label'])
    test_features_df = pd.DataFrame(test_features, columns=['feature', 'class_label'])
    train_features_df.to_json("./processed_dataset/train_data.json")
    test_features_df.to_json("./processed_dataset/test_data.json")

    print(len(train_features_df), "개의 train파일들의 특징 추출이 완료되었습니다.")
    print(len(test_features_df), "개의 test파일들의 특징 추출이 완료되었습니다.")