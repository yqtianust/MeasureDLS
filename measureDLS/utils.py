import torch
import torch.nn as nn
import torchvision
import numpy as np


class PytorchFCNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.relu(self.layer1(x))
        return self.layer2(output)

PATH_SAMPLE_PYTORCH_FC_NN = 'model.pt'

def generate_sample_PytrochFCNN():

    # Model parameters
    input_size, hidden_size, output_size = 784, 64, 10
    model = PytorchFCNN(input_size, hidden_size, output_size)

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

    '''
    2. Train with MNIST (working)
    3. Store the trained model for re-usage (working)
    '''
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

def load_sample_PytrochFCNN():
    input_size, hidden_size, output_size = 784, 64, 10
    model = PytorchFCNN(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(PATH_SAMPLE_PYTORCH_FC_NN))
    return model.eval()

