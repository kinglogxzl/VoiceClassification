import librosa
from scipy.io import wavfile
import numpy as np
import cv2
import os

path = './wav_dir/'

fileList = [x for x in os.listdir(path)]
ignore_name = ('_cut','_roll','_tune','_noise')
# print(fileList)
for file_name in fileList:
    if(file_name.endswith('.wav') and not file_name.endswith(ignore_name, 0, -4)):
        y, sr = librosa.load(path+file_name)
        len_y = len(y)

        # 剪切
        wavfile.write(path + file_name[:-4] + '_cut.wav', sr, y[int(0.2 * len_y):int(0.8 * len_y) + 1])
        # 旋转
        y_roll = np.roll(y, sr * 3)
        wavfile.write(path + file_name[:-4] + '_roll.wav', sr, y_roll)
        # 调音
        y_tune = cv2.resize(y, (1, int(len(y) * 1.2))).squeeze()
        lc = len(y_tune) - len_y
        y_tune = y_tune[int(lc / 2):int(lc / 2) + len_y]
        wavfile.write(path + file_name[:-4] + '_tune.wav', sr, y_tune)
        # 加噪声
        wn = np.random.randn(len_y)
        y_noise = np.where(y != 0.0, y + 0.01 * wn, 0.0)
        wavfile.write(path + file_name[:-4] + '_noise.wav', sr, y_noise)
