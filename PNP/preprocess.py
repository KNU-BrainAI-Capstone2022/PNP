import pandas as pd
import os
import feature_extract as ft
import moviepy.editor as mp


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

    clip = audio_clip.subclip(dur, int(audio_length) - 1)
    clip.write_audiofile(output_path + '\\' + audio_id + '-' + str(num).rjust(2, '0') + '.wav')

    return num


if __name__ == "__main__":
    if not os.path.exists("./processed_dataset"):
        os.mkdir("./processed_dataset")
        os.mkdir("./processed_dataset/train")
        os.mkdir("./processed_dataset/test")

    # 각종 경로 설정
    fulldatasetpath = './musicnet'
    metadata = pd.read_csv(fulldatasetpath + '/musicnet_metadata.csv')
    bf_train_abspath = os.path.abspath(fulldatasetpath + "/train_data/")
    bf_test_abspath = os.path.abspath(fulldatasetpath + "/test_data/")
    af_train_abspath = os.path.abspath("./processed_dataset/train")
    af_test_abspath = os.path.abspath("./processed_dataset/test")

    test_id = ["1759", "1819", "2106", "2191", "2298", "2303", "2382", "2416", "2556", "2628"]

    # 최종 데이터를 모음
    train_features = []
    test_features = []
    # 각 사운드 파일별 특징 추출
    n = 0
    for index, row in metadata.iterrows():
        class_label = row["composition"]
        id_ = str(row["id"])
        if id_ in test_id:
            bf_file_name = os.path.join(bf_test_abspath, id_ + ".wav")
            n_data = split_music(bf_file_name, af_test_abspath, id_)
            for i in range(n_data + 1):
                af_file_name = os.path.join(af_test_abspath + '\\' + id_ + '-' + str(i).rjust(2, '0') + '.wav')
                data = ft.make_mfcc(af_file_name)
                # test_features.append([data, class_label])
        else:
            bf_file_name = os.path.join(bf_train_abspath, id_ + ".wav")
            n_data = split_music(bf_file_name, af_train_abspath, id_)
            for i in range(n_data + 1):
                af_file_name = os.path.join(af_train_abspath + '\\' + id_ + '-' + str(i).rjust(2, '0') + '.wav')
                data = ft.make_mfcc(af_file_name)
                # train_features.append([data, class_label])
        # print(class_label)
        # print(file_name)
        # n += 1
        # print(n, "개 추출 완료")

    # 리스트 -> 데이터 프레임 및 데이터 저장
    # features_df = pd.DataFrame(features, columns=['feature', 'class_label'])
    #features_df.to_json("./Processed Data/Processed_data.json")

    #print(len(features_df), "개의 파일들의 특징 추출이 완료되었습니다.")