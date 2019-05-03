import re
import math
from sklearn import metrics
from sklearn.metrics import accuracy_score
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
    Y_predict=[]    
    Lagrange_values=[]
    weight_opt_list=[]
    b_list=[]
    K= compute_dot_product(X,'poly_kernel')
    Lagrange_values= compute_lagrange_multipliers(X,Y,K)
    #print(len(Lagrange_values))
    weight_opt_list= compute_opt_weights(Lagrange_values,X,Y)
    #print(len(weight_opt_list))
    b_list= compute_bias(Lagrange_values,Y)
    #print(len(b_list))
    return (weight_opt_list,b_list)
#Pediction part
def svm_precit(weight_opt_list,X, b_list):
    Y_predict=[]
    pred_col=[]
    weight_opt_list=np.array(weight_opt_list)
    b_list=np.array(b_list)
    for i in range(len(X)):
        pred = (weight_opt_list[i]*np.sum(X[i,:]))+(b_list[i])
        if pred > 0:
            Y_predict.append(1)
            pred_col.append('1')
        if pred <= 0:
            Y_predict.append(0)
            pred_col.append('not-spam') 
    return(Y_predict,pred_col)