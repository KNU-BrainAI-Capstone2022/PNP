import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

max_pad_len = 2584
n_fft = 2048
win_length = 2048
hop_length = 512
sr = 22050


# 특징 추출 함수
def make_stft(file_name):
    try:
        audio, _ = librosa.load(file_name, sr=sr)
        print('-------Processing STFT-------')
        print('sr:', sr, ', audio shape:', audio.shape)
        print('length:', audio.shape[0] / float(sr), 'secs')
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        print('Shape before padding:', stft_db.shape)
        pad_width = max_pad_len - stft_db.shape[1]
        if pad_width > 0:
            stft_db = np.pad(stft_db, pad_width=((0, 0), (0, pad_width)), mode="constant")
        print('Shape after padding:', stft_db.shape)
    except Exception as e:
        print("Error encountered while parsing file:", file_name)
        return None
    return stft_db


def make_mel(file_name):
    try:
        audio, _ = librosa.load(file_name, sr=sr)
        print('-------Processing Mel-------')
        print('sr:', sr, ', audio shape:', audio.shape)
        print('length:', audio.shape[0] / float(sr), 'secs')
        mel = librosa.feature.melspectrogram(y=audio, sr=sr)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        print('Shape before padding:', mel_db.shape)
        pad_width = max_pad_len - mel_db.shape[1]
        if pad_width > 0:
            mel_db = np.pad(mel_db, pad_width=((0, 0), (0, pad_width)), mode="constant")
        print('Shape after padding:', mel_db.shape)
    except Exception as e:
        print("Error encountered while parsing file:", file_name)
        return None
    return mel_db


def make_mfcc(file_name):
    try:
        audio, _ = librosa.load(file_name, sr=sr)
        # print('-------Processing MFCC-------')
        # print('sr:', sr, ', audio shape:', audio.shape)
        # print('length:', audio.shape[0] / float(sr), 'secs')
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_fft=n_fft,
                                    hop_length=hop_length, n_mfcc=30, dct_type=2)
        # print('Shape before padding:', mfcc.shape)
        # pad_width = max_pad_len - mfcc.shape[1]
        # if pad_width > 0:
        #    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
        # print('Shape after padding:', mfcc.shape)
    except Exception as e:
        print("Error encountered while parsing file:", file_name)
        return None
    return mfcc  # 특징이 추출된 MFCC의 정보가 든 np.array를 반환한다.


def make_cqt(file_name):
    try:
        audio, _ = librosa.load(file_name, sr=sr)
        print('-------Processing CQT-------')
        print('sr:', sr, ', audio shape:', audio.shape)
        print('length:', audio.shape[0] / float(sr), 'secs')
        cqt = librosa.cqt(y=audio, sr=sr, hop_length=hop_length)
        cqt = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
        print('Shape before padding:', cqt.shape)
        pad_width = max_pad_len - cqt.shape[1]
        if pad_width > 0:
            cqt = np.pad(cqt, pad_width=((0, 0), (0, pad_width)), mode="constant")
        print('Shape after padding:', cqt.shape)
    except Exception as e:
        print("Error encountered while parsing file:", file_name)
        return None
    return cqt


"""
# 각 추출 방식으로 그래프 그려서 비교
fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(8, 9))
data1 = make_stft('C:/Users/user/PycharmProjects/PNP/processed_dataset/train/1727-07.wav')
data2 = make_mel('C:/Users/user/PycharmProjects/PNP/processed_dataset/train/1727-07.wav')
data3 = make_cqt('C:/Users/user/PycharmProjects/PNP/processed_dataset/train/1727-07.wav')
data4 = make_mfcc('C:/Users/user/PycharmProjects/PNP/processed_dataset/train/1727-07.wav')
img1 = librosa.display.specshow(data1, hop_length=hop_length, x_axis='time', y_axis='log', ax=ax[0])
ax[0].set(title='STFT (log scale)')
img2 = librosa.display.specshow(data2, hop_length=hop_length, x_axis='time', y_axis='mel', ax=ax[1])
ax[1].set(title='Mel-Spectrogram')
img3 = librosa.display.specshow(data3, hop_length=hop_length, x_axis='time', y_axis='cqt_hz', ax=ax[2])
ax[2].set(title='Constant-Q')
img4 = librosa.display.specshow(data4, hop_length=hop_length, x_axis='time', ax=ax[3])
ax[3].set(title='MFCC')

for ax_i in ax:
    ax_i.label_outer()

fig.colorbar(img1, ax=[ax[0], ax[1], ax[2]], format="%+2.f dB")
fig.colorbar(img4, ax=[ax[3]])
# ax[0].set(xlim=[1, 3])  # 1-3초 부분만 확대
plt.show()
"""
