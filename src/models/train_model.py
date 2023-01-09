from models.CNN import Network
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

model = Network()  # this is our LightningModule
early_stopping_callback = EarlyStopping(
  monitor="val_loss", patience=3, verbose=True, mode="min"
)  
checkpoint_callback = ModelCheckpoint(dirpath = "./models")

trainer = Trainer(callbacks=[checkpoint_callback, early_stopping_callback], max_epochs=30, gpus=1)
trainer.fit(model)