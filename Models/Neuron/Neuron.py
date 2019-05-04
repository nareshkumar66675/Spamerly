import numpy as np
import sys
import pandas as pd
import os.path
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from collections import defaultdict
from NeuralArch.NeuralNet import *
import pickle
import seaborn as sns;
sns.set()

import matplotlib.pylab as plt

def GetData():
    testFilePath = r"C:\Users\kumar\OneDrive\Desktop\spam_processed_data.csv\spam_processed_data.csv"
    rawDF = pd.read_csv(testFilePath)

    # Reorder Decision Attribute for consistency
    textDF = rawDF.iloc[:,6:]
    textDF.loc[:,len(rawDF.columns)] = rawDF.iloc[:,2]

    return textDF


def GetMushroomData():
    testFilePath = r"C:\Users\kumar\OneDrive\Documents\Projects\Neuron\DataSet\agaricus-lepiota.data"
    rawDF = pd.read_csv(testFilePath,header=None)

    # Reorder Decision Attribute for consistency
    mushroomDF = rawDF.iloc[:,1:]
    mushroomDF.loc[:,len(rawDF.columns)] = rawDF.iloc[:,0]

    return mushroomDF

def GetCarData():
    testFilePath = r"C:\Users\kumar\OneDrive\Documents\Projects\Neuron\DataSet\car.data"
    rawDF = pd.read_csv(testFilePath)

    ## Reorder Decision Attribute for consistency
    #mushroomDF = rawDF.iloc[:,1:]
    #mushroomDF.loc[:,len(encodedDF.columns)] = rawDF.iloc[:,0]

    return rawDF

def GetLearningRate(selectedDF,X_train,X_test,y_test ):
    # return 0.3
    lRateAccuracy = {}
    for lRate in range(1,10):
            net = NeuralNet(np.size(X_train,1)-1,1000,1,2)

            model = net.trainModel(X_train.tolist(),lRate/10,20)

            pred = net.testModel(X_test.tolist(), model)

            accuracy = net.calculateAccuracy(y_test.tolist(),pred)

            print("For Learning Rate : {0} the Prediction Rate is {1}%".format(lRate/10,"{0:.2f}".format(accuracy)))

            lRateAccuracy[lRate/10] = accuracy

    lists = sorted(lRateAccuracy.items()) # sorted by key, return a list of tuples

    x, y = zip(*lists) # unpack a list of pairs into two tuples

    #plt.plot(x, y)

    sns.lineplot(x=x, y=y)
    plt.show()

    return max(lRateAccuracy, key=lRateAccuracy.get)

def GetOptimalEpoch(selectedDF,X_train,X_test,y_test,lRate ):

    #return 9
    epochAccuracy = {}
    for epoch in range(3,20):
            net = NeuralNet(len(selectedDF.columns)-1,len(selectedDF.columns)-1,1,len(selectedDF.iloc[:,-1].unique()))

            model = net.trainModel(X_train.values,lRate,epoch)

            pred = net.testModel(X_test.values, model)

            accuracy = net.calculateAccuracy(y_test.values,pred)

            print("For Epoch : {0} the Prediction Rate is {1}%".format(epoch,"{0:.2f}".format(accuracy)))

            epochAccuracy[epoch] = accuracy

    lists = sorted(epochAccuracy.items()) # sorted by key, return a list of tuples

    x, y = zip(*lists) # unpack a list of pairs into two tuples

    #plt.plot(x, y)

    sns.lineplot(x=x, y=y)
    plt.show()

    return max(epochAccuracy, key=epochAccuracy.get)

def GetOptimalHiddenLayers(selectedDF,X_train,X_test,y_test,lRate,epoch ):
    hiddenLayerAccuracy = {}
    for hiddenLayer in range(1,6):
            net = NeuralNet(len(selectedDF.columns)-1,len(selectedDF.columns)-1,hiddenLayer,len(selectedDF.iloc[:,-1].unique()))

            model = net.trainModel(X_train.values,lRate,epoch)

            pred = net.testModel(X_test.values, model)

            accuracy = net.calculateAccuracy(y_test.values,pred)

            print("For Optimal Layer Count : {0} the Prediction Rate is {1}%".format(hiddenLayer,"{0:.2f}".format(accuracy)))

            hiddenLayerAccuracy[hiddenLayer] = accuracy

    lists = sorted(hiddenLayerAccuracy.items()) # sorted by key, return a list of tuples

    x, y = zip(*lists) # unpack a list of pairs into two tuples

    #plt.plot(x, y)

    sns.lineplot(x=x, y=y)
    plt.show()

    return max(hiddenLayerAccuracy, key=hiddenLayerAccuracy.get)


testFilePath = r"/Users/sandips/Desktop/Spamerly/Models/Neuron/DataSet/SMSSpamCollection"
textDF = pd.read_csv(testFilePath,sep='\t',header=None)

## Reorder Decision Attribute for consistency
#textDF = rawDF.iloc[:,6:]
#textDF.loc[:,len(rawDF.columns)] = rawDF.iloc[:,2]


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(textDF[1]).toarray()


from sklearn import preprocessing

genreLabel = preprocessing.LabelEncoder()
genreLabel.fit(textDF[0])

textDF[0]= genreLabel.transform(textDF[0]) 

#textDF = GetData()

X_train, X_test, y_train, y_test = train_test_split(
features, textDF[0], random_state=42)



y_trainShaped = y_train.values.reshape(y_train.size,1)
X_train = np.concatenate((X_train, y_trainShaped), 1)

#X_train.loc[:,len(textDF.columns)] = y_train
#X_test.loc[:,len(textDF.columns)] = y_test

y_testShaped = y_test.values.reshape(y_test.size,1)
X_test = np.concatenate((X_test, y_testShaped), 1)



#print("Finding Best Optimal Learning Rate")
#optimalLearningRate = GetLearningRate(textDF,X_train, X_test,y_test )
#print("Optimal Learning Rate '{0}'".format(optimalLearningRate))


net = NeuralNet(np.size(X_train,1)-1,1500,1,2)

model = net.trainModel(X_train.tolist(),0.1,100)

pickle.dump(model, open("save1.p", "wb"))

#with open('save.p', 'rb') as handle:
#    model = pickle.load(handle)

#pred = net.testModel(X_test[:2,:].tolist(), model)
pred = net.testModel(X_test.tolist(), model)



#print(pred)

#accuracy = net.calculateAccuracy(y_test[:2].tolist(),pred)
accuracy = net.calculateAccuracy(y_test.tolist(),pred)

 print("Prediction Rate '{0}'".format("{0:.2f}".format(accuracy)))

print(accuracy)

#while True:
#    print("Select DataSet")
#    print("1. Car DataSet")
#    print("2. Mushroom Dataset")

#    dataChoice = int(input('Select one Dataset from above : '))

#    if dataChoice == 1:
#        selectedDF = GetCarData()
#    elif dataChoice == 2:
#        selectedDF = GetMushroomData()
#    else:
#        choice = input('Enter Valid Option. Press Y to Restart and N to Exit: ')
#        if str.lower(choice) == 'n':
#            sys.exit()
#        else:
#            continue


#    dfLabelEncoder = defaultdict(preprocessing.LabelEncoder)
#    # Encoding the variable
#    fit = selectedDF.apply(lambda x: dfLabelEncoder[x.name].fit_transform(x))

#    # Inverse the encoded
#    #fit.apply(lambda x: d[x.name].inverse_transform(x))

#    # Using the dictionary to label future data
#    selectedDF = selectedDF.apply(lambda x: dfLabelEncoder[x.name].transform(x))

    
#    min_max_scaler = preprocessing.MinMaxScaler()
#    np_scaled = min_max_scaler.fit_transform(selectedDF.iloc[:,0:len(selectedDF.columns)])
#    df_normalized = pd.DataFrame(np_scaled)

#    X_train, X_test, y_train, y_test = train_test_split(
#    df_normalized, selectedDF.iloc[:,-1], random_state=42)


#    X_train.loc[:,len(selectedDF.columns)] = y_train
#    X_test.loc[:,len(selectedDF.columns)] = y_test


#    #net = NeuralNet(len(selectedDF.columns)-1,10,1,len(selectedDF.iloc[:,-1].unique()))

#    #model = net.trainModel(X_train.values,0.3,20)

#    #pred = net.testModel(X_test.values, model)

#    #accuracy = net.calculateAccuracy(y_test.values,pred)

#    #print("Prediction Rate '{0}'".format("{0:.2f}".format(accuracy)))

#    #print("Prediction Rate" + accuracy)



#    #for i in lRateAccuracy:
#    #    print(i, lRateAccuracy[i])
#    print("Finding Best Optimal Learning Rate")
#    optimalLearningRate = GetLearningRate(selectedDF,X_train, X_test,y_test )
#    print("Optimal Learning Rate '{0}'".format(optimalLearningRate))


#    print("Finding Best Epoch Value")
#    epochValue = GetOptimalEpoch(selectedDF,X_train, X_test,y_test,optimalLearningRate )
#    print("Optimal Epoch Value '{0}'".format(epochValue))

#    print("Finding Optimal no of Hidden layers")
#    hiddenLayerCount = GetOptimalHiddenLayers(selectedDF,X_train, X_test,y_test,optimalLearningRate,epochValue )
#    print("Optimal Hidden Layer Count '{0}'".format(hiddenLayerCount))
    


#print("End")



