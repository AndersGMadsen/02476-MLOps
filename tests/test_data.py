from tests import _PATH_DATA
from src.data.data_utils import CorruptMNIST

def test_data():
    N_train = 25000 #40000
    N_test = 5000

    train = CorruptMNIST(root_dir=_PATH_DATA+'/raw', train=True, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=self.batch_size, num_workers=6)

    
    assert len(dataset) == N_train for training and N_test for test
    assert (datapoint.shape == [1,28,28] for datapoint in dataset)
    assert that all labels are represented
