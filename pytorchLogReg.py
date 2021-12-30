from flwr.client.app import start_numpy_client
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

from torchvision import models
from torchsummary import summary 

import time

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #SEES GPU BABY!
print("is the GPU available? ->", torch.cuda.is_available())

# read all of the csv files for patient activity data, processedPlus has been modified to have label numbers
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

#Attach all participant data into a list
myData = [part1, part2, part3, part4, part5, part6, part7, part8, part9, part10]

# Set the column/feature names
colNames = ['accel-x', 'accel-y', 'accel-z', 'linear-x', 'linear-y', 'linear-z', 'gyro-x', 'gyro-y', 'gyro-z', 'mag-x', 'mag-y', 'mag-z', 'activity', 'labelNumber']
features = ['accel-x', 'accel-y', 'accel-z', 'linear-x', 'linear-y', 'linear-z', 'gyro-x', 'gyro-y', 'gyro-z', 'mag-x', 'mag-y', 'mag-z']
label = ['labelNumber'] 

#For each participant, splits data 80% trainset and 20% testset
splitData = prepData.splitIndividualDatas(myData)

#Final split has data for each participant in form ([train_X, train_y, test_X, test_y])
finalSplit = prepData.finalPrep(splitData, features, label)
print(finalSplit[0][1])
print(type(finalSplit[0][1]))
print(type(finalSplit[0][1][1]))
#print(finalSplit[0]) #finalSplit[0] is participant 1 [0]->train_X, [1]->train_y, [2]->test_X, [3]->test_y

#pools all training data and corresponding testing data
pooledTrainingData = prepData.poolData(finalSplit) 
pooledTrainingData = [pooledTrainingData[0], np.array(pooledTrainingData[1]), pooledTrainingData[2], np.array(pooledTrainingData[3])]
pooledTensor_X = torch.tensor(pooledTrainingData[0].values.astype(np.float32))
# print("type of train matrix:")
# print(type(pooledTensor_X))
pooledTensor_y = torch.tensor(pooledTrainingData[1].astype(np.int64))
# print("type of label array")
# print(type(pooledTensor_y))
pooledTensor_X_test = torch.tensor(pooledTrainingData[2].values.astype(np.float32))
pooledTensor_y_test = torch.tensor(pooledTrainingData[3].astype(np.int64))

#Turns pooled dataframe into tensor !CENTRALIZED DATA TO TRAIN!
tensorPooled = [pooledTensor_X, pooledTensor_y, pooledTensor_X_test, pooledTensor_y_test] 
#print(len(tensorPooled[0]), len(tensorPooled[1]), len(tensorPooled[2]), len(tensorPooled[3]))
#Convert panda dataframes into tensors that work in pytorch. !INDIVIDUAL DATA TO TRAIN!
tensorSplit = prepData.framesToTensors(finalSplit) 
# print(tensorSplit[0][0].shape)
# print(tensorSplit[0][1].shape)
# hyper parameters used for training.
batch_size = 64
input_size = 1*12
num_classes = 7

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #tester...
# print("Is my gpu available... ", torch.cuda.is_available())
#CREATE THE MODEL
class ActivityModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, output_size) #what is this 8 supposed to be? #changed 8->outputsize
        #self.drop = nn.Dropout(0.1) #I believe that this dropout layer is making my accuracy fluctuate??.... comment out....
        #self.linear2 = nn.Linear(8, output_size) # output size is going to be vector of 7 values (*one val per activity) ##commented out 
        #input sizes feature size, output size corresponds to the number of classes as logistic regression
        # returns probability corresponding to each class
    
    #returns output obtained after the input data has been passed though layers of model
    def forward(self, x):
        outputs = self.linear1(x)
        #x2 = self.drop(x1)  #COMMENT OUT TO TEST IF This will stop the accuracy from fluctuating so much...
        #outputs = self.linear2(x1) #CHANGED x2 to x1 ##commented out...
        return outputs
    
    """
    Every input in the given batch is passed through the forward function that can be called
    within its class as 'self'. The outputs are then compared with the targets to calculate and return
    the loss. We use the cross-entropy loss as this is a classification problem.
    """
    def training_step(self, batch):
        inputs, targets = batch
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE) #possible gpu speed up...
        #print("Currently using... ", DEVICE) #tester
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, targets)
        return loss
    
    """
    Inputs are passed though the forward function to obtain outputs used to calculate cross-entropy loss.
    Accuracy is calculated as the number of correct predictions in the batch divided by the total
    number of predictions done. The .detach() is required to set requires_grad = False so that
    the values are excluded from gradient calculation. 
    """
    def validation_step(self, batch):
        inputs, targets = batch
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE) #possible gpu speed up...
        outputs = self(inputs)
        print("outputs: ")
        print(outputs)
        loss = F.cross_entropy(outputs, targets)
        _, pred = torch.max(outputs, 1)
        print("pred: ")
        print(pred)
        accuracy = torch.tensor(torch.sum(pred==targets).item()/len(pred))
        return [loss.detach(), accuracy.detach()]

### function that evaluates the performance of the model ERROR in this FUNCTION....
def evaluate(model, loader):
    outputs = [model.validation_step(batch) for batch in loader] 
    outputs = torch.tensor(outputs).T
    loss, accuracy = torch.mean(outputs, dim=1)
    return loss, accuracy  ##returns the loss and accuracy for 391 entries not entire test dataset..

#Function that fits the data to the model
def fit(model, train_loader, val_loader, epochs, lr, optimizer_function = torch.optim.Adam):  #I should really fit based on percentage of matrix size
    history = {"loss" : [], "accuracy": []}
    optimizer = optimizer_function(model.parameters(), lr)
    for epoch in range(epochs):
        #print("Epoch ", epoch)
        #Train
        start_time = time.time()
        for batch in train_loader: #swapping zero grad to be before loss.backward
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        #Validate
        #for batch in val_loader: #I do not evalute model in validate loop //error in this code batch then batch again...
        #Don't really need to test/validate after each epoch...
        loss, accuracy = evaluate(model, val_loader)#evaluate is not modifying state dict
        # if(epoch % 10 == 0):
        print("-----%s seconds ---" % (time.time() - start_time)) #for time of one epoch
        #time of GPU execution -> -----3.5995371341705322 seconds ---
        #time of CPU execution ->-----1.374497652053833 seconds ---
        #     print("Epoch:", epoch, "loss: ", loss.item(), "accuracy: ", accuracy.item(), "\n")
        history['loss'].append(loss.item()) #one epoch takes 6 seconds on cpu....
        history['accuracy'].append(accuracy.item())

        
    return history

testModel = ActivityModel(12, 7)
summary(testModel, (1, 12))

## time of GPU one participant -> -----190.27161955833435 seconds ---
## time of CPU one participant -> -----135.46855449676514 seconds ---

# SETTING THE DATALOADERS FOR A SINGLE PARTICIPANT
# train_data = TensorDataset(tensorSplit[0][0], tensorSplit[0][1])
# test_data = TensorDataset(tensorSplit[0][2], tensorSplit[0][3])
# train_loader = DataLoader(train_data, 5000, shuffle = True)
# val_loader = DataLoader(test_data, 300)

 #train 7200 test 500 20 epochs accuracy 78.8
  #train 5000 test 300 20 epochs accuracy 84.5
#Do I really need both a test and validation dataset?

# print(tensorSplit[0][2].shape[0])

# ## MODEL AND FIT TO START TRAINING AND VALIDATION
# model = ActivityModel(12, 7)
# history = fit(model, train_loader, val_loader, 20, 0.01)
# print("loss after X epochs: ", history['loss'][-1], " accuracy: ", history['accuracy'][-1])

#loss, accuracy = evaluate(model, test_loader)
#print("Evaluation result: Loss: ", loss.item(), " Accuracy; ", accuracy.item())

#worry about validation later? -> question for professor M


def individualTraining(tensorSplit):
    counter = 1
    for participant in tensorSplit:
        train_data = TensorDataset(participant[0], participant[1])
        test_data = TensorDataset(participant[2], participant[3])
        train_loader = DataLoader(train_data, 64, shuffle = True) #testing with 64
        val_loader = DataLoader(test_data, 32, shuffle=True) #batch size = 1
        model = ActivityModel(12, 7)
        model.to(DEVICE)
        start_time = time.time()
        history = fit(model, train_loader, val_loader, 100, 0.005) 
        print("-----%s seconds ---" % (time.time() - start_time)) #for time of one participant fitting model
        print("Participant ", counter, " loss: ", history['loss'][-1], " accuracy: ", history['accuracy'][-1])
        counter +=1

def individualOnPooled(tensorSplit, pooledData): ##working on this #10:45 on GPU
    counter = 1
    for participant in tensorSplit:
        train_data = TensorDataset(participant[0], participant[1])
        # print(participant[0].shape)
        # print(participant[1].shape)
        # print(participant[1][1])
        test_data = TensorDataset(pooledData[2], pooledData[3])
        train_loader = DataLoader(train_data, 64, shuffle = True) #testing with 64
        val_loader = DataLoader(test_data, 32, shuffle=True) #batch size = 1
        model = ActivityModel(12, 7)
        history = fit(model, train_loader, val_loader, 100, 0.01) ##error with history I'm not working with history anymore? lr .01 ->.005
        print("Participant ", counter, " loss: ", history['loss'][-1], " accuracy: ", history['accuracy'][-1]) ##evaluation looks correct...
        counter +=1

#requires tensorPooled
def centralTraining(tensorSplit, tensorPooled):
    #I just need to train the model 1 with the pooled data then I can just evaluate for accuracy. 
    #I fit the model with the pooled training data and set participant 1 test as validation set, I don't actually want to look at it's accuracy.
    train_data = TensorDataset(tensorPooled[0], tensorPooled[1])
    valid_data = TensorDataset(tensorSplit[0][2], tensorSplit[0][3]) #validate with participant 1's test data
    train_loader = DataLoader(train_data, 64, shuffle=True) #15% as batch size 72000 Adam works 64 Adam doesn't work
    val_loader = DataLoader(valid_data, 32, shuffle=True) 
    model = ActivityModel(12, 7)
    #model.to(DEVICE)
    history = fit(model, train_loader, val_loader, 100, 0.01)
    counter = 1
    for participant in tensorSplit:
        particpantTest = TensorDataset(participant[2], participant[3])
        test_loader = DataLoader(particpantTest, 32, shuffle=True) ##I need the tensorDataset first
        loss, accur = evaluate(model, test_loader)
        print("Participant ", counter, " loss: ", loss.item(), " accuracy: ", accur.item())
        counter +=1

#requires tensorPooled
def centralPooledTest(tensorPooled):
    #I just need to train the model 1 with the pooled data then I can just evaluate for accuracy. 
    #I fit the model with the pooled training data and set participant 1 test as validation set, I don't actually want to look at it's accuracy.
    train_data = TensorDataset(tensorPooled[0], tensorPooled[1])
    valid_data = TensorDataset(tensorPooled[2], tensorPooled[3]) 
    train_loader = DataLoader(train_data, 64, shuffle=True) #15% as batch size 72000 Adam works 64 Adam doesn't work
    val_loader = DataLoader(valid_data, 32, shuffle=True) 
    model = ActivityModel(12, 7)
    #model.to(DEVICE)
    history = fit(model, train_loader, val_loader, 100, 0.01)

    loss, accur = evaluate(model, val_loader)
    print("Centralized on Pooled Test Data -> loss: ", loss.item(), " accuracy: ", accur.item())


"""
Something odd is still going on in this code....
I need to figure out how to attach pytorch to my GPU so I don't have to wait so long...
"""

#################Individual training 100 epochs on self data 10/30/21###############################
# working on individualTraining
# Participant  1  loss:  0.27258485555648804  accuracy:  0.9166931509971619
# Participant  2  loss:  0.9994723796844482  accuracy:  0.7099725008010864
# Participant  3  loss:  0.46520841121673584  accuracy:  0.8219120502471924
# Participant  4  loss:  0.5080676674842834  accuracy:  0.8478479385375977
# Participant  5  loss:  1.433089017868042  accuracy:  0.6700771450996399
# Participant  6  loss:  0.9298048615455627  accuracy:  0.6552188992500305
# Participant  7  loss:  0.45426708459854126  accuracy:  0.8183956742286682
# Participant  8  loss:  0.49887898564338684  accuracy:  0.8278870582580566
# Participant  9  loss:  1.110442876815796  accuracy:  0.6624101400375366
# Participant  10  loss:  0.9057558178901672  accuracy:  0.6526808142662048
#################Individual training 100 epochs 10/30/21###############################

################################# Individual on Pooled Data !!! Should be less accurate!!!!####################
# Participant  1  loss:  14.048957824707031  accuracy:  0.382729172706604
# Participant  2  loss:  3.149073839187622  accuracy:  0.43408772349357605
# Participant  3  loss:  7.094602108001709  accuracy:  0.36100971698760986
# Participant  4  loss:  1.7333972454071045  accuracy:  0.4984129071235657
# Participant  5  loss:  4.955145835876465  accuracy:  0.44568151235580444
# Participant  6  loss:  2.1306440830230713  accuracy:  0.5237985849380493
# Participant  7  loss:  6.532655715942383  accuracy:  0.3769441843032837
# Participant  8  loss:  3.971605062484741  accuracy:  0.4506411850452423
# Participant  9  loss:  4.77351713180542  accuracy:  0.4248031973838806 
# Participant  10  loss:  2.0809261798858643  accuracy:  0.4936436712741852
################################# Individual on Pooled Data !!! Should be less accurate!!!!####################

if __name__ == '__main__':
     #print("I'm working on individualOnPooled")
     #individualOnPooled(tensorSplit, tensorPooled) ##honestly accuracie just fluctuates a lot during 100 epochs 84-90% random... 
    print("working on individualTraining")
    individualTraining(tensorSplit) #calls for individual training #I'm now using classic log reg lr.001
    #centralTraining(tensorSplit, tensorPooled) #might be an issue....
    # Maybe I should normalize my data inside of the participant csv.... !!!!1
## 2) maybe it's the mysterious dropout layer....

# print("I'm working on centralized training...")
# centralTraining(tensorSplit, tensorPooled)
#################Centralized training using individual test sets 100 epochs 10/30/21##############################
# Participant  1  loss:  0.7167047262191772  accuracy:  0.7806947827339172
# Participant  2  loss:  0.9284606575965881  accuracy:  0.6555890440940857
# Participant  3  loss:  1.0965672731399536  accuracy:  0.5792882442474365
# Participant  4  loss:  1.1990504264831543  accuracy:  0.6930520534515381
# Participant  5  loss:  0.9099887609481812  accuracy:  0.6745980978012085
# Participant  6  loss:  0.8651291131973267  accuracy:  0.7005340456962585
# Participant  7  loss:  1.08334219455719  accuracy:  0.45325717329978943
# Participant  8  loss:  0.6633076071739197  accuracy:  0.7295368313789368
# Participant  9  loss:  1.0247684717178345  accuracy:  0.6529187560081482
# Participant  10  loss:  0.8776682019233704  accuracy:  0.6716370582580566
#################Centralized training 100 epochs 10/30/21##############################

# print("Working on centralPooledTest...")
# centralPooledTest(tensorPooled)
#######################Centralized training using pooled test set 100 epochs###########################
# Centralized on Pooled Test Data -> loss:  0.2422116994857788  accuracy:  0.91651850938797
#######################Centralized training using pooled test set 100 epochs###########################

##Individual training  train 5000 test 300 epochs 20 ||time required to train model 15 minutes and 13 seconds for 10 logistic regression models 83% accurate
# Participant  1  loss:  0.5201037526130676  accuracy:  0.823888897895813
# Participant  2  loss:  0.4533368945121765  accuracy:  0.8403174877166748
# Participant  3  loss:  0.48277202248573303  accuracy:  0.845634937286377
# Participant  4  loss:  0.498978853225708  accuracy:  0.8274603486061096
# Participant  5  loss:  0.49489012360572815  accuracy:  0.826904833316803
# Participant  6  loss:  0.486783891916275  accuracy:  0.8126189708709717
# Participant  7  loss:  0.4886251986026764  accuracy:  0.8317460417747498
# Participant  8  loss:  0.5085568428039551  accuracy:  0.8295238018035889
# Participant  9  loss:  0.49023592472076416  accuracy:  0.8396031260490417
# Participant  10  loss:  0.4894466996192932  accuracy:  0.8329364061355591

##Individual training  train 7200 test 500 epochs 20 ||time required to train model 10 minutes and 07 seconds for 10 logistic regression models 78% accurate
# Participant  1  loss:  0.602836549282074  accuracy:  0.7831538319587708
# Participant  2  loss:  0.5630471110343933  accuracy:  0.7940000295639038
# Participant  3  loss:  0.636439323425293  accuracy:  0.7680768966674805
# Participant  4  loss:  0.5854231715202332  accuracy:  0.788153886795044
# Participant  5  loss:  0.6067163944244385  accuracy:  0.7760000228881836
# Participant  6  loss:  0.5975051522254944  accuracy:  0.7807692289352417
# Participant  7  loss:  0.6548885703086853  accuracy:  0.7663846611976624
# Participant  8  loss:  0.609332799911499  accuracy:  0.7866923213005066
# Participant  9  loss:  0.5811623334884644  accuracy:  0.8089999556541443
# Participant  10  loss:  0.5828595757484436  accuracy:  0.7886923551559448

##Individual training  train 64 test 500 epochs 20 
# Participant  1  loss:  0.3513783812522888  accuracy:  0.8732539415359497
# Participant  2  loss:  0.3154054880142212  accuracy:  0.8895238041877747
# Participant  3  loss:  0.4184189438819885  accuracy:  0.8419047594070435
# Participant  4  loss:  0.3315909802913666  accuracy:  0.879444420337677
# Participant  5  loss:  0.32928985357284546  accuracy:  0.8825396299362183
# Participant  6  loss:  0.4143725037574768  accuracy:  0.8365079164505005
# Participant  7  loss:  0.436903715133667  accuracy:  0.8635715246200562
# Participant  8  loss:  0.32876014709472656  accuracy:  0.8832539916038513
# Participant  9  loss:  0.32845771312713623  accuracy:  0.8948412537574768
# Participant  10  loss:  0.40242254734039307  accuracy:  0.8600000739097595

##Centralized training ||time required to train model 2 minutes and 37 seconds for 1 logistic regression models 79% accurate 20 epochs optimizer Adam
# Participant  1  loss:  0.6202189922332764  accuracy:  0.7933076620101929
# Participant  2  loss:  0.6334702372550964  accuracy:  0.7837691903114319
# Participant  3  loss:  0.6242830753326416  accuracy:  0.7913846373558044
# Participant  4  loss:  0.6286941170692444  accuracy:  0.7884615659713745
# Participant  5  loss:  0.6243686079978943  accuracy:  0.7938461899757385
# Participant  6  loss:  0.6262997388839722  accuracy:  0.7896922826766968
# Participant  7  loss:  0.6311438083648682  accuracy:  0.7859230637550354
# Participant  8  loss:  0.6265965700149536  accuracy:  0.7880769371986389
# Participant  9  loss:  0.6294581890106201  accuracy:  0.786230742931366
# Participant  10  loss:  0.6296812295913696  accuracy:  0.7903845906257629

# #Centralized training ||time required to train model 2 minutes and 37 seconds for 1 logistic regression models 79% accurate 20 epochs optimizer SGD
# Participant  1  loss:  1.029807209968567  accuracy:  0.6166154146194458
# Participant  2  loss:  1.0356911420822144  accuracy:  0.6118461489677429
# Participant  3  loss:  1.031514048576355  accuracy:  0.6179999709129333
# Participant  4  loss:  1.0327935218811035  accuracy:  0.6153076887130737
# Participant  5  loss:  1.0368574857711792  accuracy:  0.6128461360931396
# Participant  6  loss:  1.0359132289886475  accuracy:  0.6198461055755615
# Participant  7  loss:  1.0263175964355469  accuracy:  0.6267691850662231
# Participant  8  loss:  1.0293247699737549  accuracy:  0.6190768480300903
# Participant  9  loss:  1.0376620292663574  accuracy:  0.6148461699485779
# Participant  10  loss:  1.0375773906707764  accuracy:  0.615538477897644

#Central train 64 Adam train
# Participant  1  loss:  47.51322937011719  accuracy:  0.3523845970630646
# Participant  2  loss:  47.67340087890625  accuracy:  0.3558461368083954
# Participant  3  loss:  47.62666320800781  accuracy:  0.35138460993766785
# Participant  4  loss:  47.53282165527344  accuracy:  0.35507693886756897
# Participant  5  loss:  47.66445541381836  accuracy:  0.3537692427635193
# Participant  6  loss:  47.77571105957031  accuracy:  0.3546923100948334
# Participant  7  loss:  47.46739196777344  accuracy:  0.35461539030075073
# Participant  8  loss:  47.51010513305664  accuracy:  0.3566923141479492
# Participant  9  loss:  47.51271057128906  accuracy:  0.35523074865341187
# Participant  10  loss:  47.47447204589844  accuracy:  0.35292309522628784

#Central train 72000 Adam 
# Participant  1  loss:  0.5418803095817566  accuracy:  0.8115383982658386
# Participant  2  loss:  0.5422793030738831  accuracy:  0.8146153688430786
# Participant  3  loss:  0.5444427132606506  accuracy:  0.8156153559684753
# Participant  4  loss:  0.5412177443504333  accuracy:  0.8153846263885498
# Participant  5  loss:  0.5370521545410156  accuracy:  0.8155384063720703
# Participant  6  loss:  0.5409154295921326  accuracy:  0.8165384531021118
# Participant  7  loss:  0.5372282266616821  accuracy:  0.8173846006393433
# Participant  8  loss:  0.5397946238517761  accuracy:  0.817384660243988
# Participant  9  loss:  0.539129376411438  accuracy:  0.8179998993873596
# Participant  10  loss:  0.5431570410728455  accuracy:  0.8119999766349792

#Central train 64 SGD
# Participant  1  loss:  3.5010359287261963  accuracy:  0.3233846127986908
# Participant  2  loss:  3.5031933784484863  accuracy:  0.3228461444377899
# Participant  3  loss:  3.5094821453094482  accuracy:  0.32253843545913696
# Participant  4  loss:  3.5011162757873535  accuracy:  0.32384616136550903
# Participant  5  loss:  3.5063230991363525  accuracy:  0.32230764627456665
# Participant  6  loss:  3.500260591506958  accuracy:  0.32230764627456665
# Participant  7  loss:  3.5012331008911133  accuracy:  0.32423079013824463
# Participant  8  loss:  3.5088918209075928  accuracy:  0.32238462567329407
# Participant  9  loss:  3.499086856842041  accuracy:  0.3236153721809387
# Participant  10  loss:  3.5002236366271973  accuracy:  0.32330769300460815

#Central UPDATED 64 Train Adam
# Participant  1  loss:  0.32009491324424744  accuracy:  0.8903846740722656
# Participant  2  loss:  0.3144453763961792  accuracy:  0.8905385136604309
# Participant  3  loss:  0.30998626351356506  accuracy:  0.8924615383148193
# Participant  4  loss:  0.30803436040878296  accuracy:  0.8916922807693481
# Participant  5  loss:  0.309997022151947  accuracy:  0.8936154246330261
# Participant  6  loss:  0.3102143704891205  accuracy:  0.8949230909347534
# Participant  7  loss:  0.3088480234146118  accuracy:  0.8933846354484558
# Participant  8  loss:  0.3083961308002472  accuracy:  0.8943846225738525
# Participant  9  loss:  0.3104969561100006  accuracy:  0.892307698726654
# Participant  10  loss:  0.31448739767074585  accuracy:  0.892230749130249