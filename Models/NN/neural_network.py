from sklearn import preprocessing
import numpy as np
from math import exp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import math

# Sigmoid's derivative function
def derivative(x):
    return x * (1.0 - x)

# Sigmoid Function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

X = []
Y = []

textDF = pd.read_csv("SMSSpamCollection",sep='\t',header=None)

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(textDF[1]).toarray()


genreLabel = preprocessing.LabelEncoder()
genreLabel.fit(textDF[0])

textDF[0]= genreLabel.transform(textDF[0]) 

X_train, X_test, Y_train, y_test = train_test_split(features, textDF[0], random_state=42)


X = X_train
y = Y_train

Y_train = Y_train.values.reshape(Y_train.size,1)
y_test  = y_test.values.reshape(y_test.size,1)


dim1 = len(X_train[0])
print("Feature Size : ", dim1)
dim2 = 4


np.random.seed(1)
weight0 = 2 * np.random.random((dim1, dim2)) - 1
weight1 = 2 * np.random.random((dim2, 1)) - 1


for j in xrange(250):

    layer_0 = X_train
    layer_1 = sigmoid(np.dot(layer_0,weight0))
    layer_2 = sigmoid(np.dot(layer_1,weight1))


    layer_2_error = Y_train - layer_2


    layer_2_delta = layer_2_error * derivative(layer_2)
    layer_1_error = layer_2_delta.dot(weight1.T)
    layer_1_delta = layer_1_error * derivative(layer_1)


    weight1 += layer_1.T.dot(layer_2_delta)
    weight0 += layer_0.T.dot(layer_1_delta)



layer_0 = X_test
layer_1 = sigmoid(np.dot(layer_0,weight0))
layer_2 = sigmoid(np.dot(layer_1,weight1))
correct = 0


# if the output is > 0.5, then label as spam else no spam
for i in xrange(len(layer_2)):
    if(layer_2[i][0] > 0.5):
        layer_2[i][0] = 1
    else:
        layer_2[i][0] = 0


    if(layer_2[i][0] == y_test[i][0]):
        correct += 1


print("Total : ", len(layer_2))
print("Correct  : ", correct)
print("Accuracy : ", round(correct * 100.0 / len(layer_2)))


# text = ["Oh k...i'm watching here"]
# text = ["07732584351 - Rodger Burns - MSG = We tried to call you re your reply to our sms for a free nokia mobile + free camcorder. Please call now 08000930705 for delivery tomorrow"]
text = ["As a valued customer"]
tfidf = TfidfVectorizer(sublinear_tf=True, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
Xtext = tfidf.fit_transform(text).toarray()

# print(len(weight0))
# print(weight0)
# print(weight1)
# print(len(weight1))

layer_0x = Xtext

# print(Xtext)
# print(len(layer_0x[0]))


# sliced_weight = weight0[:len(layer_0x[0]),:]

# layer_1x = sigmoid(np.dot(layer_0x,sliced_weight))
# layer_2x = sigmoid(np.dot(layer_1x,weight1))

layer_1x = sigmoid(np.dot(layer_0x,weight0))
layer_2x = sigmoid(np.dot(layer_1x,weight1))

# def truncate(n, decimals=0):
#     multiplier = 10 ** decimals
#     return int(n * multiplier) / multiplier


# # if the output is > 0.5, then label as spam else no spam
# if(layer_2x[0][0] > 0.5):
#     print("Spam")
# else:
#     print("Not Spam")
print("Result : ", (layer_2x[0][0],80))
