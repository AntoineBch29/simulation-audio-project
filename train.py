from Data.Datamodule.Datamodule import MyDataModule
import lightning as L

from Model.LightningModule import Module

if __name__ == "__main__":
    model = Module()
    datamodule = MyDataModule(batch_size=2,num_workers=1)

    # Init trainer
    trainer = L.Trainer(
        max_epochs=3,
        accelerator="auto",
        devices=1,
    )
    # Pass the datamodule as arg to trainer.fit to override model hooks :)
    trainer.fit(model, datamodule)



