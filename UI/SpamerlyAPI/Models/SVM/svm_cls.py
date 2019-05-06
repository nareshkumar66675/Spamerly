import re
import math
import matplotlib.pyplot as plt
import numpy as np
from cvxopt import matrix, solvers
import preprocess_train as pre
def linear_kernel(x, y):
    x = np.array(x)
    y = np.array(y)
    return np.dot(x, y)
def poly_kernel(x, y, d = 4):
    x = np.array(x)
    y = np.array(y)
    return (np.dot(x, y) + 1) ** d

def rbf_kernel(x, y, sigma = 1):
    x = np.array(x)
    y = np.array(y)
    norm = np.linalg.norm(x - y)
    return np.exp(- (norm ** 2) / (sigma ** 2))
def compute_dot_product(X,kernel):
        n_samples, n_features = X.shape
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
             for j in range(n_samples):
                    if kernel=="linear_kernel":
                            K[i,j] = linear_kernel(X[i,:], X[j,:])
                    if kernel=="poly_kernel":
                            K[i,j] = poly_kernel(X[i,:], X[j,:])
                    if kernel=="rbf_kernel":
                            K[i,j] = rbf_kernel(X[i,:], X[j,:])
        return(K)
def compute_lagrange_multipliers(X,Y,K):
        C=None
        show_progress = False
        # Setup solver inputs
        data_size = Y.shape[0]
        P = matrix(K, tc='d') # d means floats
        q = matrix(np.full(Y.shape, -1, dtype=float), tc='d')
        G = matrix(-np.identity(data_size), tc='d') if C is None \
            else matrix(np.concatenate((-np.identity(data_size), np.identity(data_size))), tc='d')
        b = matrix(np.zeros(1), tc='d')
        A = matrix(Y, tc='d').T
        h = matrix(np.zeros(data_size), tc='d') if C is None \
            else matrix(np.concatenate((np.zeros(data_size),C * np.ones(data_size))), tc='d')   
        solvers.options['show_progress'] = show_progress
        solution = solvers.qp(P, q, G, h, A, b)['x'] # Get oprtimal values
        return np.asarray(solution).reshape((data_size,)) # Convert matrix to numpy array
    
def compute_opt_weights(Lagrange_values,X,Y):
    weight_opt_list=[]
    for i in range(len(Lagrange_values)):
        sample_weight = Lagrange_values[i]*Y[i]*np.sum(X[i,:])
        weight_opt_list.append(sample_weight)
    return(weight_opt_list)

def compute_bias(Lagrange_values,Y):
    b_list=[]
    for i in range(Y.shape[0]):
        sample_b= Lagrange_values[i]*Y[i]
        b_list.append(sample_b)
    return(b_list)
#X:
#[X1...........X1i] [Yi]
#[x1,x2,x3.....xn][y1]
#[x1,x2,x3.....xn][y2]
#[x1,x2,x3.....xn][yn]
def svm_training(X,Y):
  #  clf.fit(X,Y)
    Y_predict=[]    
    Lagrange_values=[]
    weight_opt_list=[]
    b_list=[]
    K= compute_dot_product(X,'linear_kernel')
    Lagrange_values= compute_lagrange_multipliers(X,Y,K)
    #print(len(Lagrange_values))
    weight_opt_list= compute_opt_weights(Lagrange_values,X,Y)
    #print(len(weight_opt_list))
    b_list= compute_bias(Lagrange_values,Y)
    #print(len(b_list))
    return (weight_opt_list,b_list)
#Pediction part for Support vector machines
def svm_precit(weight_opt_list,X, b_list):
    Y_predict=[]
    pred_col=[]
    Y_prob=[]
    weight_opt_list=np.array(weight_opt_list)
    b_list=np.array(b_list)
    #new_opt_weight =np.sum(weight_opt_list)
 #   for i in range(len(X)):
    #pred= clf.predict(X)
    prev_weight=weight_opt_list[0]
    for weight in weight_opt_list:
        if (weight>0) and (weight <prev_weight):
                    min_weight=weight
        prev_weight=weight
    prev_b=b_list[0]
    for b in b_list:
        if (b>0) and (b <prev_b):
                    min_b=b
        prev_b=b   
    print(min_b)     
    print(min_weight)
    print(X)
    print(np.sum(X))
    pred = (min_weight*np.sum(X))+(min_b) #pred = np.sum(weight_opt_list)*np.sum(X)+np.sum(b_list)
    if pred > 0:
            if pred <=0.5:
                Y_prob.append(2)
            else:
                Y_prob.append(3)
            Y_predict.append(1)
            pred_col.append(pred)
    if pred <= 0:
            if pred >(-0.5):
                Y_prob.append(1)
            else:
                Y_prob.append(0)
            Y_predict.append(0)
            pred_col.append(pred) 
    return(Y_predict,pred_col,Y_prob)