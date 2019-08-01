import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np

PATH_SAMPLE_PYTORCH_FC_MNIST = 'tests/sample_model_PYTORCH_FC_MNIST.pt'
PATH_SAMPLE_PYTORCH_CNN_MNIST = 'tests/sample_model_PYTORCH_CNN_MNIST.pt'
PATH_SAMPLE_PYTORCH_FC_CIFAR10 = 'tests/sample_model_PYTORCH_FC_CIFAR10.pt'
PATH_SAMPLE_PYTORCH_CNN_CIFAR10 = 'tests/sample_model_PYTORCH_CNN_CIFAR10.pt'


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


class PytorchFC_CIFAR10(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(3*32*32, 64)
        self.layer2 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.relu(self.layer1(x))
        return self.layer2(output)


class PytorchCNN_CIFAR10(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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

    torch.save(model.state_dict(), PATH_SAMPLE_PYTORCH_FC_MNIST)


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

    torch.save(model.state_dict(), PATH_SAMPLE_PYTORCH_CNN_MNIST)


def generate_sample_Pytroch_FC_CIFAR10():

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True)

    model = PytorchFC_CIFAR10()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    num_of_epochs, num_of_print_interval = 1, 2000
    for epoch in range(num_of_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            inputs = inputs.reshape(-1, 3*32*32)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % num_of_print_interval == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    torch.save(model.state_dict(), PATH_SAMPLE_PYTORCH_FC_CIFAR10)


def generate_sample_Pytroch_CNN_CIFAR10():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True)

    model = PytorchCNN_CIFAR10()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    num_of_epochs, num_of_print_interval = 5, 2000
    for epoch in range(num_of_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % num_of_print_interval == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    torch.save(model.state_dict(), PATH_SAMPLE_PYTORCH_CNN_CIFAR10)


def load_sample_Pytroch_FC_MNIST():
    input_size, hidden_size, output_size = 784, 64, 10
    model = PytorchFC_MINST(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(PATH_SAMPLE_PYTORCH_FC_MNIST))
    return model


def load_sample_Pytroch_CNN_MNIST():
    num_of_channels, width, height = 1, 28, 28
    model = PytorchCNN_MNIST(num_of_channels, width, height)
    model.load_state_dict(torch.load(PATH_SAMPLE_PYTORCH_CNN_MNIST))
    return model


def load_sample_Pytroch_FC_CIFAR10():
    model = PytorchFC_CIFAR10()
    model.load_state_dict(torch.load(PATH_SAMPLE_PYTORCH_FC_CIFAR10))
    return model


def load_sample_Pytroch_CNN_CIFAR10():
    model = PytorchCNN_CIFAR10()
    model.load_state_dict(torch.load(PATH_SAMPLE_PYTORCH_CNN_CIFAR10))
    return model
