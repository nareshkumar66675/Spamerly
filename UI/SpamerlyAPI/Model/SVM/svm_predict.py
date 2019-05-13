import os
import nltk
from nltk import word_tokenize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import math
import matplotlib.pyplot as plt
import numpy as np
from cvxopt import matrix, solvers
np.random.seed(6)
import pickle
import Model.SVM.preprocess_train as pre
import Model.SVM.svm_cls as svm1
#Imported SVM to compare the results with Sklearn model
from sklearn import svm
clf = svm.SVC(kernel='linear', C = 1.0,probability=True)
list1=[]
nltk.download('stopwords')
nltk.download('punkt')
def svm_model(text):
    word_occurance_dict={}
    word_attributes_list=[]
    l=[]
    C=None
    show_progress = False
    Y_predict=[]
    pred_col=[]
    weight_opt_list=np.loadtxt(open("weight_opt_list.csv", "rb"), delimiter=",")
    b_list=np.loadtxt(open("b_list.csv", "rb"), delimiter=",") 
    res2= pd.read_csv('features.csv')
    word_attributes_list=res2
    test_df=pre.preprocess_test_data(text)
    full_test_df= pre.set_test_word_attributes(test_df,word_attributes_list)
    X1 =test_df[test_df.columns[4:]]
    X1=np.array(X1)
    Y_predict1,pred_col1,Y_prob=svm1.svm_precit(weight_opt_list,X1, b_list)
    #print(Y_predict1,pred_col1,Y_prob)
 #   full_test_df['Y_predict'] =Y_predict1
 #   full_test_df['pred_col'] =pred_col1
 #   full_test_df['Y_prob']=Y_prob
    return (Y_predict1,Y_prob1)
#Predcition usin sklearn SVM model to verify and comare the prediction results

clf,word_attributes_list1 = pickle.load(open( "svmModel.p", "rb" ))
def svm_model_new(text):
    #sev_val=0.0
    #df=pd.read_csv(r"C:\Users\kumar\OneDrive\Documents\Projects\Spamerly\UI\SpamerlyAPI\Model\SVM\SMSSpamCollection.csv")
    #ind_list,spam_count,non_spam_count= pre.create_dataframe(df)
    #df['Ind']= ind_list
    #df['Text'] = [entry.lower() for entry in df['Text']]
    #df['Text'].dropna(inplace=True)
    #df['tokenized'] =df['Text'].apply(word_tokenize)
    #item_list= pre.remove_stop_special(df)
    #df['tokenized_text']=item_list
    #item_list= pre.remove_stop_special(df)
    #df['tokenized_text']=item_list
    #word_list=pre.get_uniques_spam(df)
    #word_occurance_dict=pre.get_word_occurance(word_list)
    #df['Label'] = df['Label'].map( {'spam': 1, 'ham': 0} ).astype(int)
    #IG_dict=pre.get_Information_gain(spam_count,non_spam_count, word_occurance_dict,df)
    #train_df=pre.set_word_attributes(IG_dict,df)
    #word_attributes_list=list(train_df.columns.values)
    #word_attributes_list1=word_attributes_list[5:]
    #features = np.array(word_attributes_list1)
    #l= train_df['Label'].tolist()
    #Y = np.array(l)
    #X =train_df[train_df.columns[5:]]
    #X=np.array(X)
    #clf.fit(X,Y)
    #pickle.dump([clf,word_attributes_list1],open( "svmModel.p", "wb" ) )
    test_df=pre.preprocess_test_data(text)



    #g=0
    #prob=0
    #for i in test_df['tokenized_text']:
    #    for j in i:
    #       if j in word_attributes_list1:
    #           g=g+1
    #prob=(g/spam_count)
    #if prob > 0.005:
    #    sev_val=3
    #elif prob < 0.005 and prob >=0.004:
    #    sev_val=2
    #elif prob < 0.003 and prob>=0.002:
    #    sev_val=1
    #elif prob < 0.002:
    #    sev_val=0
    full_test_df= pre.set_test_word_attributes(test_df,word_attributes_list1)
    X1 =test_df[test_df.columns[4:]]
    X1=np.array(X1)
    
    #pred=clf.predict(X1)
    pred = clf.predict_proba(X1)
    print(clf.classes_)
    prob = pred[0][1]
    #if prob > 0.1:
    #    sev_val=3
    #elif prob < 0.1 and prob >=0.02:
    #    sev_val=2
    #elif prob < 0.02 and prob>=0.002:
    #    sev_val=1
    #elif prob < 0.002:
    #    sev_val=0

    if prob > 0.6:
        sev_val=3
    elif prob < 0.6 and prob >=0.4:
        sev_val=2
    elif prob < 0.4 and prob>=0.2:
        sev_val=1
    else:
        sev_val=0

    return sev_val,round(prob,2)