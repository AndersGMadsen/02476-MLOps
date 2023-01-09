import argparse
import sys

from torch.utils.data import DataLoader
import torch as nn
import click

from src.data.data_utils import mnist
from models.model import Network

from tqdm import tqdm

@click.group()
def cli():
    pass

@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
@click.option("--epochs", default=50, help='number of epochs')
def train(lr, epochs):
    train_dataset, _ = mnist()
    
    trainloader = DataLoader(train_dataset, batch_size=1000, shuffle=True)

    device = nn.device("cuda:0")

    model = Network().to(device)

    criterion = nn.nn.CrossEntropyLoss()
    optimizer = nn.optim.Adam(model.parameters(), lr=lr)
    
    steps = 0
    running_loss = 0

    pbar = tqdm(total = epochs)

    for e in range(epochs):
        # Model in training mode, dropout is on
        model.train()
        for images, labels in trainloader:
            steps += 1

            images = images.to(device)
            labels = labels.to(device)
            
            # Flatten images into a 784 long vector
            optimizer.zero_grad()

            output = model.forward(images)

            loss = criterion(output, labels.flatten())
        
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        pbar.set_description("Loss: {:.3f}".format(loss.item()))
        pbar.update(1)
    
    nn.save(model.state_dict(), 'checkpoint.pth')

@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    criterion = nn.nn.CrossEntropyLoss()
    _, test_dataset = mnist()
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    checkpoint = nn.load(model_checkpoint)
    model = Network()
    model.load_state_dict(checkpoint)

    accuracy = 0
    test_loss = 0
    for images, labels in testloader:
        output = model.forward(images)
        #print(output)
        test_loss += criterion(output, labels.flatten()).item()

        ## Calculating the accuracy 
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = nn.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        accuracy += (labels.data == ps.max(1)[1])

    
    print(test_loss, (accuracy / len(test_dataset)).item())

cli.add_command(train)
cli.add_command(evaluate)

if __name__ == "__main__":
    cli()


    
    
    
    