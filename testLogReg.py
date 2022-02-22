import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing as prep
from sklearn.preprocessing import StandardScaler

# read all of the csv files for patient activity data
part1 = pd.read_csv('processedData/Participant_1.csv')
part2 = pd.read_csv('processedData/Participant_2.csv')
part3 = pd.read_csv('processedData/Participant_3.csv')
part4 = pd.read_csv('processedData/Participant_4.csv')
part5 = pd.read_csv('processedData/Participant_5.csv')
part6 = pd.read_csv('processedData/Participant_6.csv')
part7 = pd.read_csv('processedData/Participant_7.csv')
part8 = pd.read_csv('processedData/Participant_8.csv')
part9 = pd.read_csv('processedData/Participant_9.csv')
part10 = pd.read_csv('processedData/Participant_10.csv')

# print(part1)
# print(type(part1))  #each patient data is a pandas dataframe

allData = [part1, part2, part3, part4, part5, part6, part7, part8, part9, part10]

# Set the column/feature names
colNames = ['accel-x', 'accel-y', 'accel-z', 'linear-x', 'linear-y', 'linear-z', 'gyro-x', 'gyro-y', 'gyro-z', 'mag-x', 'mag-y', 'mag-z', 'activity']
features = ['accel-x', 'accel-y', 'accel-z', 'linear-x', 'linear-y', 'linear-z', 'gyro-x', 'gyro-y', 'gyro-z', 'mag-x', 'mag-y', 'mag-z']
label = ['activity']

for data in allData:
    data.columns = colNames


# Train 10 different models leaving one participant as test data (maybe 2 participants?)
def createTestingArray(allData, testIndex):

    trainingData = pd.DataFrame();
    for index in range(0, len(allData)):
        if(index == testIndex):
            continue
        elif(trainingData.empty):
            trainingData = allData[index]
        else:
            trainingData = trainingData.append(allData[index], ignore_index=True)

    X_train = trainingData[features]
    y_train = trainingData[label].values.flatten()
    X_test = allData[testIndex][features]
    y_test = allData[testIndex][label].values.flatten()

    #normalize all data by features
    # X_train = prep.normalize(X_train, axis=0)
    # X_test = prep.normalize(X_test, axis=0)


    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test) #I don't use fit on test data, I don't want to generate learning model parameters


    return[X_train, y_train, X_test, y_test]

#firstTestArray = createTestingArray(allData, 0)  #firstTestArray shows I can create model data and label, test data and labels

#Method for finding the accuracy based on logistic regression
def findAccuracy(allData, numbParticipation):

    ## Creates and empty dataframe for my info
    data = {'Participant-Number': [],
            'Accuracy' : [],
            'Precision' : []
            }
    
    statsFrame = pd.DataFrame(data)

    for participant in range(0, numbParticipation):
        testArray = createTestingArray(allData, participant)
        #logreg = LogisticRegression() #defualt solver is lbfgs #lbfgs is better for smaller datasets so I should use saga as solver.
        logreg = LogisticRegression(solver='saga')
        logreg.fit(testArray[0], testArray[1])
        y_pred = logreg.predict(testArray[2])

        accur = metrics.accuracy_score(testArray[3], y_pred)
        precision = metrics.precision_score(testArray[3], y_pred, average="weighted")

        newRow = {'Participant-Number':participant, 'Accuracy' : accur, 'Precision' : precision} 
        statsFrame = statsFrame.append(newRow, ignore_index=True)

    return statsFrame

def centralizedTraining(pooledDataSet, splitDataSet):
    ## Creates and empty dataframe for my info
    data = {'Participant-Number': [],
            'Accuracy' : [],
            'Precision' : []
            }
    statsFrame = pd.DataFrame(data)

    #keeps track of participant number
    partNumber = 0

    #classifier should be initialized before loop
    classifier = LogisticRegression(solver='saga') #struggles with lbfgs
    classifier.fit(pooledDataSet[0], pooledDataSet[1])

    for participant in splitDataSet:
        partNumber += 1

        # Prediction is made based on a participant's test_X
        y_pred = classifier.predict(participant[2])

        accur = metrics.accuracy_score(participant[3], y_pred)
        precision = metrics.precision_score(participant[3], y_pred, average="weighted")

        newRow = {'Participant-Number':partNumber, 'Accuracy': accur, 'Precision' : precision}
        statsFrame = statsFrame.append(newRow, ignore_index=True)

    return statsFrame

# Trains models based on individual user's data
def individualTraining(splitDataSet):
    ## Creates and empty dataframe for my info
    data = {'Participant-Number': [],
            'Accuracy' : [],
            'Precision' : []
            }
    statsFrame = pd.DataFrame(data)

    #keeps track of participant number
    partNumber = 0

    for participant in splitDataSet:
        partNumber += 1

        #classifier should be initialized inside loop
        classifier = LogisticRegression(solver='saga')
        classifier.fit(participant[0], participant[1])

        # Prediction is made based on a participant's test_X
        y_pred = classifier.predict(participant[2])

        accur = metrics.accuracy_score(participant[3], y_pred)
        precision = metrics.precision_score(participant[3], y_pred, average="weighted")

        newRow = {'Participant-Number':partNumber, 'Accuracy': accur, 'Precision' : precision}
        statsFrame = statsFrame.append(newRow, ignore_index=True)

    return statsFrame



#calls and prints the final predictions
# finalPredictions = findAccuracy(allData, 10)
# print(finalPredictions)

#accuracy with normalize previous
#accuracy is not great
#    Participant-Number  Accuracy  Precision
# 0                 0.0  0.542834   0.566523
# 1                 1.0  0.480849   0.450633
# 2                 2.0  0.487516   0.475579
# 3                 3.0  0.558914   0.542097
# 4                 4.0  0.543056   0.580954
# 5                 5.0  0.587708   0.584319
# 6                 6.0  0.443388   0.449963
# 7                 7.0  0.578104   0.594066
# 8                 8.0  0.542993   0.609238
# 9                 9.0  0.486055   0.534975

#accuracy with preprocessing StandardScalar 
#    Participant-Number  Accuracy  Precision
# 0                 0.0  0.574231   0.547214
# 1                 1.0  0.560850   0.563409
# 2                 2.0  0.596549   0.614776
# 3                 3.0  0.609962   0.549213
# 4                 4.0  0.620391   0.619081
# 5                 5.0  0.623280   0.635523
# 6                 6.0  0.519135   0.502637
# 7                 7.0  0.667265   0.681642
# 8                 8.0  0.571088   0.585197
# 9                 9.0  0.626677   0.629303
