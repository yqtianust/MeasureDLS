import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

PATH_SAMPLE_PYTORCH_FC_NN = 'sample_model_PYTORCH_FC_NN.pt'
PATH_SAMPLE_PYTORCH_CNN_NN = 'sample_model_PYTORCH_CNN_NN.pt'

class PytorchFC_MINST(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.relu(self.layer1(x))
        return self.layer2(output)

class PytorchCNN_MNIST(nn.Module):
    def __init__(self, num_of_channels, width, height):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(3*3*64, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1, 3*3*64)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def generate_sample_Pytroch_FC_MINST():

    # Model parameters
    input_size, hidden_size, output_size = 784, 64, 10
    model = PytorchFC_MINST(input_size, hidden_size, output_size)

    # Optimizer parameters
    loss_func = nn.CrossEntropyLoss()
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training parameters
    num_of_epochs, num_of_print_interval = 1, 25
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=100, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Training
    total_step = len(train_loader)
    for epoch in range(num_of_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)

            # Forwarding
            outputs = model.forward(images)
            loss = loss_func(outputs, labels)

            # Optimization (back-propogation)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % num_of_print_interval == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch +
                                                                         1, num_of_epochs, i+1, total_step, loss.item()))

    torch.save(model.state_dict(), PATH_SAMPLE_PYTORCH_FC_NN)


def generate_sample_Pytroch_CNN_MNIST():

    # Model parameters
    num_of_channels, width, height = 1, 28, 28
    model = PytorchCNN_MNIST(num_of_channels, width, height)

    # Optimizer parameters
    loss_func = nn.CrossEntropyLoss()
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training parameters
    num_of_epochs, num_of_print_interval = 1, 25
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=100, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Training
    total_step = len(train_loader)
    for epoch in range(num_of_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forwarding
            outputs = model.forward(images)
            loss = loss_func(outputs, labels)

            # Optimization (back-propogation)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % num_of_print_interval == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch +
                                                                         1, num_of_epochs, i+1, total_step, loss.item()))

    torch.save(model.state_dict(), PATH_SAMPLE_PYTORCH_CNN_NN)

def load_sample_Pytroch_FC_MNIST():
    input_size, hidden_size, output_size = 784, 64, 10
    model = PytorchFC_MINST(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(PATH_SAMPLE_PYTORCH_FC_NN))
    return model.eval()


def load_sample_Pytroch_CNN_MNIST():
    num_of_channels, width, height = 1, 28, 28
    model = PytorchCNN_MNIST(num_of_channels, width, height)
    model.load_state_dict(torch.load(PATH_SAMPLE_PYTORCH_CNN_NN))
    return model.eval()

