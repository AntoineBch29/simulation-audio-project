from pathlib import Path
import sys
import torch

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
dataset = DatasetLibrispeech()

print(len(dataset))