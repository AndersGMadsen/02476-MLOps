from os.path import join

import numpy as np
import torch as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


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
        label = nn.tensor([label])

        if self.transform:
            return self.transform(img[0]), label
        else:
            return nn.from_numpy(img), label

def process_dataset(dir):
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(0.13207851, 0.30989197)])
                                
    train_data = CorruptMNIST(root_dir = dir, train = True, transform = transform)
    test_data = CorruptMNIST(root_dir = dir, train = False, transform = transform)

    train_data, train_label = next(iter(DataLoader(train_data, batch_size=len(train_data))))
    test_data, test_label = next(iter(DataLoader(test_data, batch_size=len(test_data))))

    processed_dataset = {"train": (train_data, train_label),
                        "test": (test_data, test_label)}

    return processed_dataset
