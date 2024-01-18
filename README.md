# simulation-audio-project
deep learning project 


## SELECTION DU MODELE

vous pouvez sélectionner votre modèle en changeant  la ligne 14 dans Model\LightningModule.py par le modèle souhaité: 

- self.model = UNet()
- self.model = CNN()

## TRAINING

Ensuite il suffit de lancer le training en utilisant la commande dans le terminal: python train.py
Dans train.py:
vous pouvez régler la taille des batch, et le nombre d'epoch

## TEST

 sélectionner le dernier checkpoint enregistré dans Logs. Pour cela, écrivez le path à la ligne 8 comme ceci:
-  model = Module.load_from_checkpoint("Logs/lightning_logs/version_3/checkpoints/epoch=4-step=325.ckpt",save=True)

## REGARDER LES RESULTATS

vous pouvez voir les courbes du suivi de l'apprentissages (notamment la loss function) à l'aide de tensorboard en utilisant la commande suivante:

- tensorboard --logdir Logs/...

suivi du nom où a été enregistré vos données 
    