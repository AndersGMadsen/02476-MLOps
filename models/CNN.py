import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.data_utils import CorruptMNIST


class Network(LightningModule):
    def __init__(self, batch_size, lr, data_dir):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout1d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        self.criterion = nn.CrossEntropyLoss()
        self.dataset = CorruptMNIST

        self.batch_size = batch_size
        self.lr = lr
        self.data_dir = data_dir

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx) :
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target.flatten())

        acc = (target.flatten() == preds.argmax(dim=-1)).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)

        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target.flatten())
        self.log_dict({'val_loss': loss})

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(0.13207851, 0.30989197)])

        dataset = CorruptMNIST(root_dir=self.data_dir, train=True, transform=transform)

        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True, num_workers=6)

        return loader

    def val_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(0.13207851, 0.30989197)])
        dataset = CorruptMNIST(root_dir=self.data_dir, train=False, transform=transform)
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, num_workers=6)
        
        return loader