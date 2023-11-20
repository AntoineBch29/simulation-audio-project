import lightning as L
from Model.LightningModule import Module
from Data.Datamodule.Datamodule import MyDataModule


if __name__ == "__main__":
    tb_logger = L.pytorch.loggers.TensorBoardLogger(save_dir="Logs/")
    model = Module.load_from_checkpoint("Logs/lightning_logs/version_3/checkpoints/epoch=4-step=325.ckpt",save=True)
    datamodule = MyDataModule(batch_size=4,num_workers=2)
    # Init trainer
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="auto",
        devices=1,
        logger=tb_logger,
    )
    # Pass the datamodule as arg to trainer.fit to override model hooks :)
    trainer.test(model,datamodule)