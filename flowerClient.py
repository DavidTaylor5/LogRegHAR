from collections import OrderedDict
from flwr import client

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

import flwr as fl 


"""
This is my first trial with flower as a module for federated learning. I hope it works. 
Example consists of one server and two clients all having the same model. Clients are 
responsible for generating individual weight-updates for the model based on their local
datasets. These updates are then sent to the server which will aggregate them to produce a 
better model. Finally, the server sends this improved version of the model back to each client.
A complete cycle of weight updates is called a round (communication round?).
"""

#Define device allocation in PyTorch with
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("is the GPU available? ->", torch.cuda.is_available())

"""CIFAR10, a popular colored image classification dataset for ML. PyTorch DataLoader()
downloads the training and test data that are then normalized.

Normalization helps get data within a range and reduces the skewness which helps learn 
faster and better. Normalization can also tackle the diminishing and exploding gradients 
problems. pixels [0,255], want values that mean(0.0) Standard Deviation (1.0)
"""

def load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10(".", train=True, download=True, transform=transform)
    testset = CIFAR10(".", train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32)
    return trainloader, testloader

"""Define the loss and optimizer with PyTorch. The training of the dataset is done by looping
over the dataset, measure the corresponding loss and optimize it."""
def train(net, trainloader, epochs):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE) ## why what? #maybe tells PyTorch to specifically use a certain device to work with the data
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

"""Define the validation of the machine learning network. We loop over the test set and 
measure the loss and accuracy of the test set."""
def test(net, testloader): #some of this code is inconsistent with train function
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1) #grabs the largest value and returns hot label
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return loss, accuracy

"""Functions fo Flower Clients. The Flower clients will use a simple CNN."""
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) ->torch.Tensor: #what does this function header mean
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#load model and data
net = Net()
trainloader, testloader = load_data()

"""
DEFINE THE FLOWER INTERFACE.
The flower server interacts with the clients through an interface called Client. When the
server selects a particular client for training, it sends training instructions over the
network. The client receives those instructions and calls one of the Client methods to run
your code (ie. to train the neural network we defined earlier).

Flower provides a convenience class called NumPyClient which makes it easier to implement the
Client interface when your workload uses PyTorch. Implementing NumPyClients usually means
defining the following methods (set_parameters is optional though)

1) get_parameters -> return model weight as a list of Numpy ndarrays
2) set_parameters -> update the local model weights with the parameters revceived from the server
3) fit -> set local model weights, train local model, receive the updated local model, receive the updated local model weights
4) evaluate -> test the local model

can be implemented in the following way...
"""
#verify if gpu is being used
class CifarClient(fl.client.NumPyClient): # assumes that clients have test/train loaders?
    def get_parameters(self):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters(), len(trainloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader) #added to show client side accuracy
        print("Loss: ", loss, " Accuracy: ", accuracy)
        return float(loss), len(testloader), {"accuracy":float(accuracy)}

# Create instance of out class CifarClient and add one line to actually run this client
fl.client.start_numpy_client("localhost:8080", client=CifarClient()) # This is the client

"""We only have to implement Client or NumPyClient and call
fl.client.start_client() or fl.client.start_numpy_client(). The string
'[::]:8080' tells the client which server to connect to. In our case we can run the
server and the client on the same machine, therefore we use '[::]:8080'. If we
run a truly federated workload with the server and clients running on different
machines, all that needs to change is the server_address we point the client at."""

### FLOWER SERVER
"""For simple workloads we can start a Flower server and leave all the configuration
possibilities at their default values. In a file named server.py import Flower and 
start the server:

import flwr as fl
fl.server.start_server(config={"num_rounds":3})

"""

#  HOW TO RUN THE STUFF
""" With both the client and the server ready, we can now run everything and see
federated learning in action. FL systems usually have a server and multiple
clients. We there fore have to start the server first:

$ python server.py

Then start running the clients in different terminals. Open a new termina
and start the first client.

$ python client.py

oOpen another terminal and start the second client:
$python client.py

"""





