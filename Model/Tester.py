import torch
from class_transform import retour
from scipy.io.wavfile import write
import os
import numpy as np

class tester(object):
    def __init__(self, log_path,save = False):
        self.log_path = log_path
        self.save = save
    def __call__(self,batch,mask_out):
        snr = torch.zeros(batch["Waveform"].shape[0])
        snr_gain = torch.zeros(batch["Waveform"].shape[0])
        snr_th = torch.zeros(batch["Waveform"].shape[0])
        snr_gain_th = torch.zeros(batch["Waveform"].shape[0])
        for i_batch in range(batch["Waveform"].shape[0]):
            curr_path = os.path.join(self.log_path,f"{batch['Speaker_ID'][i_batch]}_{batch['Chapter_ID'][i_batch]}_{batch['Utterance_ID'][i_batch]}")
            os.makedirs(curr_path,exist_ok=True)
            output_spectro = batch["stft"][i_batch]*(mask_out[i_batch]>0.5)
            output = retour(output_spectro,len(batch["Waveform"][i_batch]))
            Waveform_phase = batch["stft"][i_batch]*(batch["y"][i_batch]>0.5)
            Waveform_phase = retour(Waveform_phase,len(batch["Waveform"][i_batch]))
            if self.save:
                np.save(os.path.join(curr_path ,'Output.npy'),output_spectro)
                np.save(os.path.join(curr_path ,'Noised_Waveform.npy'),batch["x"][i_batch].numpy())
                np.save(os.path.join(curr_path ,'y.npy'),batch["y"][i_batch].numpy())
                np.save(os.path.join(curr_path ,'mask_out.npy'),mask_out[i_batch].numpy())
                sr = batch["Sample_rate"][i_batch]
                write(os.path.join(curr_path ,'Waveform.wav'),sr,batch["Waveform"][i_batch].numpy())
                write(os.path.join(curr_path ,'Noised_Waveform.wav'),sr,batch["Noised_Waveform"][i_batch].numpy())
                write(os.path.join(curr_path ,'Output.wav'),sr,output[0])
                write(os.path.join(curr_path ,'Waveform_phase.wav'),sr,Waveform_phase[0])
            snr[i_batch] = 10*np.log10((batch["Waveform"][i_batch].cpu()**2).sum()/((batch["Waveform"][i_batch].cpu()-output)**2).sum())
            snr_th[i_batch] = 10*np.log10((batch["Waveform"][i_batch].cpu()**2).sum()/((batch["Waveform"][i_batch].cpu()-Waveform_phase)**2).sum())
            snr_gain[i_batch] = snr[i_batch]-batch["snr"][i_batch]
            snr_gain_th[i_batch] = snr_th[i_batch]-batch["snr"][i_batch]
        return snr

class valider(object):
    def __init__(self, log_path,save = False):
        self.log_path = log_path
        self.save = save
    def __call__(self,batch,mask_out):
        snr = torch.zeros(batch["Waveform"].shape[0])
        for i_batch in range(batch["Waveform"].shape[0]):
            curr_path = os.path.join(self.log_path,f"{batch['Speaker_ID'][i_batch]}_{batch['Chapter_ID'][i_batch]}_{batch['Utterance_ID'][i_batch]}")
            output = batch["stft"][i_batch]*(mask_out[i_batch]>0.5)
            output = retour(output,len(batch["Waveform"][i_batch]))
            snr[i_batch] = 10*np.log10((batch["Waveform"][i_batch]**2).sum()/((batch["Waveform"][i_batch]-output)**2).sum())
        return snr

if __name__ == "__main__" :
    pass