import torch as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import hydra
from models.CNN import Network
#from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path='config', config_name='default_config.yaml')
def train(config):
	hparams = config.experiment
	nn.manual_seed(hparams['seed'])

	model = Network(batch_size = hparams['batch_size'], lr=hparams['lr'], data_dir=hparams['data_dir'])

	early_stopping_callback = EarlyStopping(
		monitor='val_loss', min_delta = 0.0, patience=3, verbose=True, mode='min'
	)  

	checkpoint_callback = ModelCheckpoint(dirpath = './models/checkpoints/')

	if hparams["wandb"]:
		logger = pl.loggers.WandbLogger(project='02476-mlops')
	else:
		logger = None

	trainer = Trainer(callbacks=[checkpoint_callback, early_stopping_callback], 
						logger=logger,
						max_epochs=hparams['max_epochs'], gpus=1)

	trainer.fit(model)

if __name__ == '__main__':
	train()