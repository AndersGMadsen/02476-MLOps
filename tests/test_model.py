import torch as nn
from torchvision import transforms

from models.CNN import Network
from src.data.data_utils import CorruptMNIST
from tests import _PATH_DATA


def test_model():
    transform = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize(0.13207851, 0.30989197)])
    
    data_dir = _PATH_DATA + '/raw'
    train_dataset = CorruptMNIST(root_dir=data_dir, train=True, transform=transform)

    batch_size=128
    lr=0.1

    model = Network(batch_size, lr, data_dir)
    

    # Test shape of model output, use the first sample
    assert model(train_dataset[0][0]).shape == nn.Size([1, 10])
    assert nn.sum(nn.exp(model(train_dataset[0][0]))) == 1