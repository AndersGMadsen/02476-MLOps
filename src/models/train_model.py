from models.CNN import Network
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import hydra
#from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="config", config_name="default_config.yaml")
def train(config):
	hparams = config.experiment
	
	# #torch.manual_seed(hparams["seed"])

	# model = Network()

	# early_stopping_callback = EarlyStopping(
	#   monitor="val_loss", patience=3, verbose=True, mode="min"
	# )  

	# checkpoint_callback = ModelCheckpoint(dirpath = "./models/checkpoints/")

	# trainer = Trainer(callbacks=[checkpoint_callback, early_stopping_callback], 
	#                   logger=pl.loggers.WandbLogger(project="02476-mlops"),
	#                   max_epochs=30, gpus=1)

	# trainer.fit(model)

if __name__ == "__main__":
	train()