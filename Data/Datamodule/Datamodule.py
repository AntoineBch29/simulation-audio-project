import lightning as L
import torch.utils.data as data
from Data.Datamodule.Dataset import DatasetLibrispeech
import torch

class MyDataModule(L.LightningDataModule):
    def __init__(self, batch_size, num_workers) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers


    def setup(self, stage):
        generator_determinist = torch.Generator().manual_seed(42)
        generator_random = torch.Generator()
        dataset = DatasetLibrispeech()
        train_val, self.test = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=generator_determinist)
        self.train, self.val = torch.utils.data.random_split(train_val, [0.8, 0.2], generator=generator_determinist)

    def train_dataloader(self):
        return data.DataLoader(self.train,batch_size=self.batch_size, shuffle=True,num_workers=self.num_workers)

    def val_dataloader(self):
        return data.DataLoader(self.val,batch_size=self.batch_size, shuffle=True,num_workers=self.num_workers)

    def test_dataloader(self):
        return data.DataLoader(self.test,batch_size=self.batch_size, shuffle=True,num_workers=self.num_workers)