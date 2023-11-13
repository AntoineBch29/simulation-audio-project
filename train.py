from Data.Datamodule.Datamodule import MyDataModule
import lightning as L

from Model.LightningModule import autoencoder

datamodule = MyDataModule(8,2)

# Init trainer
trainer = L.Trainer(
    max_epochs=3,
    accelerator="auto",
    devices=1,
)
# Pass the datamodule as arg to trainer.fit to override model hooks :)
trainer.fit(autoencoder, datamodule)



