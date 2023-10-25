from typing import Any
import librosa
import numpy as np
import matplotlib.pyplot as plt

class numpy_waveform(object):
    def __call__(self,sample):
        sample["Waveform"] = sample["Waveform"].numpy()[0]
        return sample
class clip_and_pad(object):
    def __init__(self,l):
        self.l = l
    def __call__(self, sample):
        
        while(len(sample["Waveform"])<self.l):
                print("coucou")
                sample["Waveform"] = np.tile(sample["Waveform"],2)
        if len(sample["Waveform"])>self.l:
            sample["Waveform"] = sample["Waveform"][:self.l]
        return sample

class stft(object):
    def __init__(self, n_fft, hop_length, win_length):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
    def __call__(self,sample):
        y_pad= librosa.util.fix_length(
        sample["Waveform"], size=len(sample["Waveform"]) + self.n_fft // 2)
        return librosa.stft(y_pad, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, sr=sample["Sample_rate"])
class fourier_signal_and_mask_power(object):
    def __init__(self,n_fft, hop_length,win_length):
        self.stft_object=stft(n_fft, hop_length,win_length)
    def __call__(self, sample):
        sample["X"]=np.log10(np.abs(self.stft_object(sample["Noised_Waveform"]))**2)
        sample["Y"]=np.abs(self.stft_object(sample["Waveform"]))**2>np.abs(self.stft_object(sample["Noise"]))**2
        return sample
class istft(object):
    def __init__(self, n_fft, hop_length, win_length):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
    def __call__(self):
        return librosa.istft(sample["Waveform"], hop_length=self.hop_length, n_fft=self.n_fft, win_length=self.win_length, length=len(sample["Waveform"]),sr=sample["Sample_rate"])
class normalize(object):
    def __call__(self,y):
        return (y-np.mean(y))/np.max(np.abs(np.min(y)),np.abs(np.max(y)))  
class noised_waveform(object):
    def __init__(self, range):
        self.range = range
    def __call__(self,sample):
        randomNums = np.random.poisson(1.5, 1)
        randomInts = np.round(randomNums)
        if randomInts<1:
            randomInts=1
        sum_alpha=0
        sample["Noise"]=librosa.load("C:/Users/pemba/Downloads/babble_16k.wav",sr=16000)
        n=np.zeros_like(sample["Noise"])
        for i in range(randomInts):
            alpha=np.random()
            sum_alpha+=alpha
            crop_noise=np.random.randint(0,len(sample["Waveform"]))
            n+=sample["Noise"][crop_noise:crop_noise+len(sample["Waveform"])]
        sample["Noise"]=n/sum_alpha    
        return sample
    
    
    
    
if __name__ == "__main__" :
    from Data.Datamodule.Dataset import DatasetLibrispeech
    dataset=DatasetLibrispeech()
    sample=dataset[0]
    y=sample["Waveform"].numpy().T
    sr=sample["Sample_rate"]
    y=y[:,0]
    p = stft(y, len(y), 2048, 1024)
    k=p()
    power_amp=power(k)
    S = power_amp()
    u, sr2 = librosa.load('./Data/Raw/babble_16k.wav',sr=16000)
    print(len(u),sr)
    p2 = stft(u, len(u), 2048, 1024)
    k2=p2()
    power_amp2=power(k2)
    S2 = power_amp2()
    m=mask(S2[:,1:94],S)
    fig, ax = plt.subplots(ncols=3)
    print("coucou")
    img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='log', x_axis='time', ax=ax[0])
    ax[0].set_title('signal ')
    fig.colorbar(img, ax=ax[0], format="%+2.0f dB")
    img = librosa.display.specshow(librosa.power_to_db(S2, ref=np.max), y_axis='log', x_axis='time', ax=ax[1])
    ax[1].set_title('noise')
    fig.colorbar(img, ax=ax[1], format="%+2.0f dB")
    img = librosa.display.specshow(librosa.power_to_db(m(), ref=np.max), y_axis='log', x_axis='time', ax=ax[2])
    ax[2].set_title('mask')
    fig.colorbar(img, ax=ax[2], format="%+2.0f dB")
    plt.show()
    j = istft(k, len(y), 2048, 1024)
    y_out = j()
    # fig=plt.figure(2)
    # randomNums = np.random.poisson(1.5, 10000)
    # randomInts = np.round(randomNums)
    # randomInts = np.clip(randomInts, 1, 10)
    # count, bins, ignored = plt.hist(randomInts, 14, density=True)
    # plt.show()
