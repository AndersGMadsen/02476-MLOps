import torch as nn
from torchvision import transforms

from src.data.data_utils import CorruptMNIST
from tests import _PATH_DATA


def test_data():

    transform = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize(0.13207851, 0.30989197)])
    
    dir = _PATH_DATA + '/raw'
    train_dataset = CorruptMNIST(root_dir=dir, train=True, transform=transform)
    test_dataset = CorruptMNIST(root_dir=dir, train=False, transform=transform)

    # Test length of data
    N_train = 25000 #40000
    N_test = 5000

    assert len(train_dataset) == N_train, "Train dataset did not have the correct number of samples"
    assert len(test_dataset) == N_test, "Test dataset did not have the correct number of samples"

    # Test shape of data
    for idx in range(len(train_dataset)):
        assert train_dataset[idx][0].shape == nn.Size([1, 28, 28]), "Train samples did not have the correct shape"
    for idx in range(len(test_dataset)):
        assert test_dataset[idx][0].shape == nn.Size([1, 28, 28]), "Test samples did not have the correct shape"

    # Test class representation
    assert nn.all(nn.unique(nn.tensor([label for image, label in train_dataset.data])) == nn.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])), "Train samples did not represent all classes"
    assert nn.all(nn.unique(nn.tensor([label for image, label in test_dataset.data])) == nn.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])), "Test samples did not represent all classes"
