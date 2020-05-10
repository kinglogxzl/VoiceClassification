import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os.path

def audio2spec(audio):

	y, sr = librosa.load(audio)
	S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
	S_dB = librosa.power_to_db(S, ref=np.max)

	plt.figure(figsize=(10, 4))
	librosa.display.specshow(S_dB, sr=sr, fmax=8000)
	plt.tight_layout()
	plt.savefig(os.path.splitext(audio)[0]+'.jpg')

	return