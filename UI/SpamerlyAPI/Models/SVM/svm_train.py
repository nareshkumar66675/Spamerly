import os
import nltk
from nltk import word_tokenize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import math
from sklearn import metrics
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from cvxopt import matrix, solvers
import csv
#Remove speacial characters
#Check stop words and special characters
np.random.seed(6)
import preprocess_train as pre
import svm_cls as svm
def main():
    word_occurance_dict={}
    l=[]
    C=None
    show_progress = False
    Y_predict=[]
    pred_col=[]
    weight_opt_list=[]
    b_list=[]
    word_occurance_dict={}
    df = pd.DataFrame(columns=['Text','Label'])
    text_list,label_list, ind_list,spam_count,non_spam_count= pre.create_dataframe()
    df['Text']=text_list
    df['Label']=label_list
    df['Ind']= ind_list
    df['Text'] = [entry.lower() for entry in df['Text']]
    df['Text'].dropna(inplace=True)
    df['tokenized'] =df['Text'].apply(word_tokenize)
    item_list= pre.remove_stop_special(df)
    df['tokenized_text']=item_list
    item_list= pre.remove_stop_special(df)
    df['tokenized_text']=item_list
    word_list=pre.get_uniques_spam(df)
    word_occurance_dict=pre.get_word_occurance(word_list)
    df['Label'] = df['Label'].map( {'spam': 1, 'not_spam': 0} ).astype(int)
    IG_dict=pre.get_Information_gain(spam_count,non_spam_count, word_occurance_dict,df)
    train_df=pre.set_word_attributes(IG_dict,df)
    word_attributes_list=list(train_df.columns.values)
    word_attributes_list=word_attributes_list[5:]
    features = np.array(word_attributes_list)
    l= train_df['Label'].tolist()
    Y = np.array(l)
    X =train_df[train_df.columns[5:]]
    X=np.array(X)
    weight_opt_list,b_list=svm.svm_training(X,Y)
    print("here")
    print(len(weight_opt_list))
    weight_df = pd.DataFrame(weight_opt_list)
    weight_df.to_csv('weight_opt_list.csv',index=False)
    b_list_df = pd.DataFrame(b_list)
    b_list_df.to_csv('b_list.csv',index=False)
  #  np.savetxt("weight_opt_list.csv", weight_opt_list, delimiter=",")
  #  np.savetxt("b_list.csv", b_list, delimiter=",")
   # np.savetxt("features.csv", features, delimiter=",")
if __name__ == '__main__':
    main()