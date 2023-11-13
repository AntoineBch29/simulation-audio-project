from pathlib import Path
import sys
import torch
from tqdm import tqdm
from scipy.io.wavfile import write

CUR_DIR_PATH = Path(__file__).resolve()
ROOT = CUR_DIR_PATH.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))# add ROOT to PATH

from Config.test import cfg

from Data.Datamodule.Dataset import DatasetLibrispeech

# dataloader = torch.utils.data.DataLoader(
#     DatasetLibrispeech,
#     batch_size=cfg["training"]["batch_size"],
#     shuffle=True,
#     num_workers=cfg["training"]["num_workers"]
#     )


generator = torch.Generator().manual_seed(42)
dataset = DatasetLibrispeech()
train_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=generator)
n = 50
print(train_set[n]["Waveform"].numpy()[0].shape)
print(len(train_set),len(test_set),len(dataset))
from class_transform import numpy_waveform,clip_and_pad
a = numpy_waveform()
b = clip_and_pad(160000)


sample = b(a(train_set[n]))
print(len(sample["Waveform"]))
write('test.wav',16000,sample["Waveform"])
# lis = []
# for i in tqdm(dataset):
#     lis.append(i["Waveform"].shape[1])

# import matplotlib.pyplot as plt
# plt.figure()
# plt.hist(lis,100)
# plt.show()