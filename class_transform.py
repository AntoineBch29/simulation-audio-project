import librosa
import numpy as np
import matplotlib.pyplot as plt


class stft(object):
    def __init__(self, y, n_y, n_fft, hop_length):
        self.y = y
        self.n_y = n_y
        self.n_fft = n_fft
        self.hop_length = hop_length
    def __call__(self):
        y_pad = librosa.util.fix_length(
            self.y, size=self.n_y + self.n_fft // 2)
        return librosa.stft(y_pad, n_fft=self.n_fft, hop_length=self.hop_length)

class power(object):
    def __init__(self, y):
        self.y = y
    def __call__(self):
        return np.abs(self.y)**2

class mask(object):
    def __init__(self, u, s):
        self.u = u
        self.s = s
    def __call__(self):
        return self.s > self.u

class log_amplitude(object):
    def __init__(self, P):
        self.P = P
    def __call__(self):
        return np.log(self.P)

class istft(object):
    def __init__(self, y, n_y, n_fft, hop_length):
        self.y = y
        self.n_y = n_y
        self.n_fft = n_fft
        self.hop_length = hop_length
    def __call__(self):
        return librosa.istft(self.y, hop_length=self.hop_length, n_fft=self.n_fft, length=self.n_y)


y, sr = librosa.load(librosa.ex("C:/Users/pemba/Documents/GitHub/Data/LibriSpeech/train-clean-100/125/121342/125-121342-0000.flac"))
p = stft(y, len(y), 2048, 1024)
k=p()
power_amp=power(k)
S = power_amp()
u, sr2 = librosa.load("C:/Users/pemba/Downloads/babble_16k.wav")
p2 = stft(u, len(u), 2048, 1024)
k2=p2()
power_amp2=power(k2)
S2 = power_amp2()
m=mask(S2[:,0:116],S)
fig, ax = plt.subplots()
img = librosa.display.specshow(librosa.power_to_db(m(), ref=np.max), y_axis='log', x_axis='time', ax=ax)
ax.set_title('Power spectrogram')
fig.colorbar(img, ax=ax, format="%+2.0f dB")
plt.show()
fig, ax = plt.subplots()
img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='log', x_axis='time', ax=ax)
ax.set_title('Power spectrogram')
fig.colorbar(img, ax=ax, format="%+2.0f dB")
plt.show()
j = istft(k, len(y), 2048, 1024)
y_out = j()

