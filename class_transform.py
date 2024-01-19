from typing import Any
import librosa
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import Compose
import torch

class numpy_waveform(object):
    def __call__(self,sample):
        sample["Waveform"] = sample["Waveform"].numpy()[0]
        return sample
class clip_and_pad(object):
    def __init__(self,l):
        self.l = l
    def __call__(self, sample):
        
        while(len(sample["Waveform"])<self.l):
                sample["Waveform"] = np.tile(sample["Waveform"],2)
        if len(sample["Waveform"])>self.l:
            sample["Waveform"] = sample["Waveform"][:self.l]
        return sample

class stft(object):
    def __init__(self, n_fft, hop_length, win_length):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
    def __call__(self,sample,name_signal):
        y_pad= librosa.util.fix_length(
        sample[name_signal], size=len(sample[name_signal]) + self.n_fft // 2)
        return librosa.stft(y_pad, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
class fourier_signal_and_mask_power(object):
    def __init__(self,n_fft, hop_length,win_length):
        self.stft_object=stft(n_fft, hop_length,win_length)
    def __call__(self, sample):
        sample["stft"] = self.stft_object(sample,"Noised_Waveform")
        sample["x"]=np.log10(np.abs(sample["stft"])**2)
        sample["y"]=(np.abs(self.stft_object(sample,"Waveform"))**2>np.abs(self.stft_object(sample,"Noise"))**2).astype(int)
        return sample
    # standardisation sur l'ensemble des données 
# class normalize(object):
#     def __call__(self,y):
#         return (y-np.mean(y))/(np.std(y))
class noised_waveform(object):
    def __init__(self, range, snr_range):
        self.range = range
        self.snr = snr_range
        self.babble=librosa.load('./Data/Raw/babble_16k.wav',sr=16000)[0]
    def __call__(self,sample):
        randomNums = np.random.poisson(self.range, 1)[0]
        randomInts = np.round(randomNums)
        if randomInts<1:
            randomInts=1
        sum_alpha=0
        sample["Noise"]=self.babble
        n=np.zeros_like(sample["Waveform"])
        for i in range(randomInts):
            alpha=np.random.rand()
            sum_alpha+=alpha
            crop_noise=np.random.randint(0,len(sample["Waveform"]))
            n+=alpha*sample["Noise"][crop_noise:crop_noise+len(sample["Waveform"])]
        snr = np.random.uniform(self.snr[0],self.snr[1])
        snr=(10**(snr/10))
        sample["snr"]=snr
        Pu=np.sum(np.abs(n/sum_alpha)**2)
        Ps=np.sum(np.abs(sample["Waveform"])**2)
        attenuation_noise=np.sqrt(Ps/(Pu*snr)) 
        sample["Noise"]=attenuation_noise*n/sum_alpha
        sample["Noised_Waveform"]=sample["Waveform"]+sample["Noise"]
        return sample
class istft(object):
    def __init__(self, n_fft, hop_length, win_length):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
    def __call__(self,sample,name_signal):
        return librosa.istft(sample[name_signal], hop_length=self.hop_length, n_fft=self.n_fft, win_length=self.win_length, length=len(sample["Waveform"]))

class to_tensor(object):
    def __call__(self, sample):
        sample["x"] = torch.from_numpy(sample["x"])
        sample["y"] = torch.from_numpy(sample["y"]).to(torch.float32)
        sample["x"] = torch.unsqueeze(sample["x"], dim=0)
        sample["y"] = torch.unsqueeze(sample["y"], dim=0)
        return sample

transforms = Compose([
        numpy_waveform(),
        clip_and_pad(160000),
        noised_waveform(range=1.5,snr_range=[5,15]),
        fourier_signal_and_mask_power(n_fft=1024, hop_length=512, win_length=1024),
        to_tensor(),
    ])
    
def retour(signal,l):
    return librosa.istft(signal.cpu().numpy(),n_fft=1024, hop_length=512, win_length=1024, length=l)

