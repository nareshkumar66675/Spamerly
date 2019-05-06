import pandas as pd 
import numpy as np 
from collections import defaultdict
import re
from sklearn.model_selection import train_test_split
#import pickle


def preprocess_text(text):

    cleaned_text=re.sub('[^a-z\s]+',' ',str(text),flags=re.IGNORECASE)
    cleaned_text=re.sub('(\s+)',' ',cleaned_text) 
    cleaned_text=cleaned_text.lower() 

    return cleaned_text 										



class BayesianAlgorithm:

    def __init__(self,unique_labels):
        
        self.class_labels=unique_labels 
        

    def addToBagofWords(self,data_values,dict_index):
        
        if isinstance(data_values,np.ndarray): data_values=data_values[0]
     
        for token_word in data_values.split(): 
          
            self.BagofWords_dicts[dict_index][token_word]+=1 
            
    def train(self,dataset,labels):

        self.data_valuess=dataset
        self.labels=labels
        self.BagofWords_dicts=np.array([defaultdict(lambda:0) for index in range(self.class_labels.shape[0])])
        
        
        if not isinstance(self.data_valuess,np.ndarray): self.data_valuess=np.array(self.data_valuess)
        if not isinstance(self.labels,np.ndarray): self.labels=np.array(self.labels)
            

        for cat_index,cat in enumerate(self.class_labels):
          
            all_cat_data_valuess=self.data_valuess[self.labels==cat] 
            
            
            cleaned_data_valuess=[preprocess_text(cat_data_values) for cat_data_values in all_cat_data_valuess]
            
            cleaned_data_valuess=pd.DataFrame(data=cleaned_data_valuess)
            
            np.apply_along_axis(self.addToBagofWords,1,cleaned_data_valuess,cat_index)
            
      
        prob_labels=np.empty(self.class_labels.shape[0])
        all_words=[]
        cat_word_counts=np.empty(self.class_labels.shape[0])
        for cat_index,cat in enumerate(self.class_labels):
           
            prob_labels[cat_index]=np.sum(self.labels==cat)/float(self.labels.shape[0]) 
            
            count=list(self.BagofWords_dicts[cat_index].values())
            cat_word_counts[cat_index]=np.sum(np.array(list(self.BagofWords_dicts[cat_index].values())))+1 
                                         
            all_words+=self.BagofWords_dicts[cat_index].keys()
                                                     

        self.vocab=np.unique(np.array(all_words))
        self.vocab_length=self.vocab.shape[0]
                                  
                                   
        denoms=np.array([cat_word_counts[cat_index]+self.vocab_length+1 for cat_index,cat in enumerate(self.class_labels)])                                                                          
      
        
        self.cats_info=[(self.BagofWords_dicts[cat_index],prob_labels[cat_index],denoms[cat_index]) for cat_index,cat in enumerate(self.class_labels)]                               
        self.cats_info=np.array(self.cats_info)                                 
                                              
                                              
    def single_text_prob(self,test_data_values):                                
                                      
                                              
        likelihood_prob=np.zeros(self.class_labels.shape[0]) 
        
        for cat_index,cat in enumerate(self.class_labels): 
                             
            for test_token in test_data_values.split(): 
                                                                     
                test_token_counts=self.cats_info[cat_index][0].get(test_token,0)+1
                                         
                test_token_prob=test_token_counts/float(self.cats_info[cat_index][2])                              
                
                likelihood_prob[cat_index]+=np.log(test_token_prob)
                                              
        post_prob=np.empty(self.class_labels.shape[0])
        for cat_index,cat in enumerate(self.class_labels):
            post_prob[cat_index]=likelihood_prob[cat_index]+np.log(self.cats_info[cat_index][1])                                  
      
        return post_prob


    def test(self,test_set):
           
       
        predictions=[] 
        for data_values in test_set: 
                                                                          
            cleaned_data_values=preprocess_text(data_values) 
                                            
            post_prob=self.single_text_prob(cleaned_data_values) 
            
            predictions.append(self.class_labels[np.argmax(post_prob)])
                
        return np.array(predictions)



def bayesian_model(text):

    training_set=pd.read_csv(r"C:\Users\kumar\OneDrive\Documents\Projects\Spamerly\UI\SpamerlyAPI\Model\SVM\SMSSpamCollection.csv") # reading the training data-set
    training_set['Label'] = training_set['Label'].map( {'spam': 0, 'ham': 1} ).astype(int)

    y_train=training_set['Label'].values
    x_train=training_set['Text'].values


    train_data,test_data,train_labels,test_labels=train_test_split(x_train,y_train,shuffle=True,test_size=0.25,random_state=42,stratify=y_train)
    labels=np.unique(train_labels)


    nb=BayesianAlgorithm(labels)
    nb.train(train_data,train_labels)

    plabels=nb.test(test_data)
    #test_acc=np.sum(plabels==test_labels)/float(test_labels.shape[0])

    #pickle.dump( nb, open( "bayesianModel.p", "wb" ) )
    # Testing on a string

    #generating predictions....
    plabels=nb.single_text_prob(text)


    if(plabels[1] > plabels[0]):
        return 0
    else:
        if(plabels[0] >= -74.59):
            return 1
        elif(plabels[0] < -74.59 and plabels[0] >= -179.59):
            return 2
        else:
            return 3


