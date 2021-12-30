import torch
import pandas as pd
#from sklearn import datasets
from torch import optim
#from torch.functional import Tensor
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, TensorDataset
import torch.nn.functional as F
import prepData
import numpy as np
import LogRegClient
import flwr as fl
import sys

from pytorchLogReg import ActivityModel

print(len(sys.argv))
print(sys.argv)
clientToRun = int(sys.argv[1]) #grabs the one command line argument and casts to int



# read all of the csv files for patient activity data
part1 = pd.read_csv('processedPlus/part1plus.csv') 
part2 = pd.read_csv('processedPlus/part2plus.csv') 
part3 = pd.read_csv('processedPlus/part3plus.csv') 
part4 = pd.read_csv('processedPlus/part4plus.csv') 
part5 = pd.read_csv('processedPlus/part5plus.csv') 
part6 = pd.read_csv('processedPlus/part6plus.csv') 
part7 = pd.read_csv('processedPlus/part7plus.csv') 
part8 = pd.read_csv('processedPlus/part8plus.csv') 
part9 = pd.read_csv('processedPlus/part9plus.csv') 
part10 = pd.read_csv('processedPlus/part10plus.csv') 


myData = [part1, part2, part3, part4, part5, part6, part7, part8, part9, part10]

# Set the column/feature names
colNames = ['accel-x', 'accel-y', 'accel-z', 'linear-x', 'linear-y', 'linear-z', 'gyro-x', 'gyro-y', 'gyro-z', 'mag-x', 'mag-y', 'mag-z', 'activity', 'labelNumber']
features = ['accel-x', 'accel-y', 'accel-z', 'linear-x', 'linear-y', 'linear-z', 'gyro-x', 'gyro-y', 'gyro-z', 'mag-x', 'mag-y', 'mag-z']
label = ['labelNumber'] #instead of ['activity']



splitData = prepData.splitIndividualDatas(myData) 

finalSplit = prepData.finalPrep(splitData, features, label) 


pooledTrainingData = prepData.poolData(finalSplit) # has a 10 participant combo of train_X and train_y in one place.
pooledTrainingData = [pooledTrainingData[0], np.array(pooledTrainingData[1])]

#print(finalSplit[0]) #finalSplit[0] is participant 1 [0]->train_X, [1]->train_y, [2]->test_X, [3]->test_y
pooledTensor_X = torch.tensor(pooledTrainingData[0].values.astype(np.float32))
pooledTensor_y = torch.tensor(pooledTrainingData[1].astype(np.int64))



tensorPooled = [pooledTensor_X, pooledTensor_y] 

# print(finalSplit[0])
# print(type(finalSplit[0][1]))
#Convert panda dataframes into tensors that work in pytorch
tensorSplit = prepData.framesToTensors(finalSplit) 

#Datasets in the form of 
def setupClients(tensorDatasets):
    clientList = []
    nameCounter = 0
    for client in tensorDatasets:
        nameCounter +=1
        user = LogRegClient.clientWithData(client[0], client[1], client[2], client[3], nameCounter)
        user.setTrainLoaders()
        clientList.append(user)
    return clientList

myClients = setupClients(tensorSplit)

# def runClients(clientList):
#     for client in clientList:
#         currUser = LogRegClient.CifarClient(ActivityModel(12, 7), client.trainloader, client.testloader)
#         fl.client.start_numpy_client("localhost:8080", client=currUser) # This is the client

# runClients(myClients)
#print(myClients[clientToRun].train_X) #same info for client 1 and 2??
#print(myClients[clientToRun].train_y)
#This code runs one of the clients, based on command line argument. 

currUser = LogRegClient.CifarClient(ActivityModel(12, 7), myClients[clientToRun].trainloader, myClients[clientToRun].testloader, myClients[clientToRun].name)
fl.client.start_numpy_client("localhost:8080", client=currUser)

#print(tensorSplit)
#print(tensorSplit[0][1]) # I now have participant data in the form (trainX trainy testX testy)


##########fix the data issues, might be relevant to pytorchLogReg..............