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
#Remove speacial characters
#Check stop words and special characters
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')
special=['!','@','#','$','%','^','&','*','(',')',':','<','>',',','.',';','{','}','|','_','-','+','=','`','~',"--","'",'"','[',']','\n','/','\\']
#special=[',','.',';','_','-','=','*']
np.random.seed(6)
def create_dataframe(df):
    ind_list=[]
    spam_count=0
    non_spam_count=0
    ind_count=0
    for i in df['Text']:
        ind_count=ind_count+1
        ind_list.append(ind_count)
    for i in df['Label']:
        if (i== str('ham')):
            non_spam_count=non_spam_count+1
        if (i== str('spam')):
            spam_count= spam_count+1
    return(ind_list,spam_count,non_spam_count)
def remove_stop_special(df):
    item_list=[]
    for item  in df['tokenized']:
        for word in item:
            word=str(word)
            if (word in stop_words) or (word in special):
                item.remove(word) 
        item_list.append(item)
    return(item_list)
def get_uniques_spam(df):
    unique_words=[]
    temp_list=[]
    for i in range(len(df.Label)):
        if(df.Label[i] == 'spam'):
            for word in df.tokenized_text[i]:
                if len(word)>= 2:
                    temp_list.append(word)
    return(temp_list)
def get_word_occurance(word_list):
    word_occurance={}
    new_word_occurance={}
    for word in word_list:
        if word in word_occurance:
            word_occurance[word] += 1
        else:
            word_occurance[word] = 1
    for k,v in word_occurance.items():
            if v>=2:
                new_word_occurance[k]=v
    return(new_word_occurance)
def get_Information_gain(spam_count,non_spam_count, word_occurance_dict,df):
    non_occurance_spam_prob_yes=0.0
    non_occurance_spam_prob_no=0.0
    occurance_spam_prob_yes=0.0
    occurance_spam_prob_no=0.0
    IG={}
    word_and_spam={}
    new_word_and_spam={}
    temp_list2=[]
    word_and_non_spam={}
    temp_list3=[]
    new_word_and_non_spam={}
    is_spam = df['Label']==1
    spam_df=df[is_spam]
    is_not_spam = df['Label']==0
    not_spam_df=df[is_not_spam]
    for item in spam_df['tokenized_text']:
        for word in item:
            temp_list2.append(word)
    for word in temp_list2:
            if word in word_and_spam:
                word_and_spam[word] += 1
            else:
                word_and_spam[word] = 1
    for k,v in  word_and_spam.items():
            #if v>=4:
                  new_word_and_spam[k]=v
    for item in not_spam_df['tokenized_text']:
        for word in item:
            temp_list3.append(word)
    for word in temp_list3:
            if word in word_and_non_spam:
                word_and_non_spam[word] += 1
            else:
                word_and_non_spam[word] = 1
    for k,v in  word_and_non_spam.items():
            #if v>=4:
                  new_word_and_non_spam[k]=v
    for k,v in word_occurance_dict.items():
    #req
        non_occurance_prob= ((len(word_occurance_dict)-v)/len(word_occurance_dict))
        if (k in word_and_spam):
            non_occurance_spam_yes= (len(word_and_spam)-word_and_spam[k])/(len(word_and_spam))
        #req
            non_occurance_spam_yes = non_occurance_spam_yes
            non_occurance_spam_prob_yes=math.log2(non_occurance_spam_yes)   
        if (k in word_and_non_spam):
            non_occurance_spam_no= (len(word_and_non_spam)-word_and_non_spam[k])/(len(word_and_spam))
        #req
            non_occurance_spam_no = non_occurance_spam_no
            non_occurance_spam_prob_no= math.log2(non_occurance_spam_no)
        #req
        occurance_prob= (v/len(word_occurance_dict))
        if (k in word_and_spam):
            occurance_spam_yes= (word_and_spam[k]/spam_count)
        #req
            occurance_spam_yes=occurance_spam_yes
            occurance_spam_prob_yes= math.log2(occurance_spam_yes)
        if (k in word_and_non_spam):
            occurance_spam_no=(word_and_non_spam[k]/len(word_and_non_spam))
        #req
            occurance_spam_no=occurance_spam_no
            occurance_spam_prob_no=math.log2(occurance_spam_no)
   
        IG[k] = ((non_occurance_prob*(non_occurance_spam_prob_yes+non_occurance_spam_prob_no))
                     +(occurance_prob*(occurance_spam_prob_yes+occurance_spam_prob_no)))
    return(IG)
def set_word_attributes(IG,df):
    sorted_IG={}
    sorted_IG = sorted(IG.items(), key=lambda kv: kv[1])
    c=0
    attri_dict={}
    for k,v in reversed(sorted_IG):
        #if c<=1000:
            attri_dict[k]=v
           # c=c+1
    att_values=[]
    for k, v in attri_dict.items():
        att_values=[]
        for item in df['tokenized_text']:
            if k in item:
                att_values.append(1)
            else:
                att_values.append(-1)
        df[k]=att_values
    return(df)
def preprocess_test_data(text):
    test_text_list=[]
    test_ind_list=[]
    test_item_list=[]
    test_df = pd.DataFrame(columns=['Text'])
    test_text_list.append(text)
    test_df['Text']=test_text_list
    test_ind_list.append(1)
    test_df['Ind']= test_ind_list
    test_df['Text'] = [entry.lower() for entry in test_df['Text']]
    test_df['tokenized'] =test_df['Text'].apply(word_tokenize)
    test_item_list= remove_stop_special(test_df)
    test_df['tokenized_text']=test_item_list
    return(test_df) 
def set_test_word_attributes(test_df,word_attributes_list):
    for feature in word_attributes_list:
        att_values1=[]
        for item in test_df['tokenized_text']:
            if feature in item:
                att_values1.append(1)
            else:
                att_values1.append(-1)
        test_df[feature]=att_values1
    return(test_df)