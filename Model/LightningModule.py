import os
from torch import optim, nn, utils, Tensor
from torchvision.transforms import ToTensor
import lightning.pytorch as pl
from Model.CNN import CNN


# define the LightningModule
class Module(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = CNN(513)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x = batch["x"]
        y = batch["y"]
        y_hat = self.model(x)
        loss = nn.functional.mse_loss(y_hat, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x = batch["x"]
        y = batch["y"]
        y_hat = self.model(x)
        loss = nn.functional.mse_loss(y_hat, y)
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer

