import torch as nn
from torch.utils.data import Dataset
import numpy as np
from os.path import join

class CorruptMNIST(Dataset):

    def __init__(self, root_dir, train, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.train = train
        self.root_dir = root_dir
        self.transform = transform

        if train:
            images = np.vstack([np.load(join(root_dir, "train_{}.npz".format(i)), allow_pickle=True)["images"] for i in range(5)]).astype(np.float32)
            labels = np.hstack([np.load(join(root_dir, "train_{}.npz".format(i)), allow_pickle=True)["labels"] for i in range(5)]).astype(np.int64)
        else:
            images = np.vstack([np.load(join(root_dir, "test.npz"), allow_pickle=True)["images"]]).astype(np.float32)
            labels = np.hstack([np.load(join(root_dir, "test.npz"), allow_pickle=True)["labels"]]).astype(np.int64)

        images = images.reshape((images.shape[0], 1, images.shape[1], images.shape[2]))
       
        self.data = list(zip(images, labels))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        img = nn.from_numpy(img)
        label = nn.tensor([label])

        if self.transform:
            return self.transform(img), label
        else:
            return img, label

def mnist():

    # exchange with the corrupted mnist dataset
    root_dir = r"C:\Users\anders\OneDrive - Danmarks Tekniske Universitet\Dokumenter\Courses\02476 Machine Learning Operations Jan\dtu_mlops\data\corruptmnist"
    train_data = CorruptMNIST(root_dir = root_dir, train = True)
    test_data = CorruptMNIST(root_dir = root_dir, train = False)

    return train_data, test_data


