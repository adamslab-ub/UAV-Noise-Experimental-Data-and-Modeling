"""

@author: Rayhaan Iqbal

Trains the machine learning model based on model type.
"""


import numpy as np
import pandas as pd
import torch.utils.data
from functools import partial
import matplotlib.pyplot as plt
import os
from network import *
#import res
import loggging
import config1 as c
import plotly
import pickle
import random
import copy
import scipy.io

def build_model(model, mat_file):

    norm =1;
#%% Dataset Generation
    
    
    # from scipy.io import loadmat
    # mat_file = loadmat('Dataset13Train_NewPartialPhys.mat')
    
    if model ==1:
        import config1 as c
        #mat_file = loadmat('NEWDataset4.mat')
        trainset_Raw_X = mat_file['train_X']
        trainset_Raw_Y = mat_file['train_Y']
        trainset_Raw = np.concatenate((trainset_Raw_X, trainset_Raw_Y), axis=1)
        
        testset_Raw_X = mat_file['test_X']
        testset_Raw_Y = mat_file['test_Y']
        testset_Raw = np.concatenate((testset_Raw_X, testset_Raw_Y), axis=1)
    elif model == 2:
        import config1 as c
        trainset_Raw_X = mat_file['train_X']
        trainset_Raw_X_PP = mat_file['train_PP_spl']
        trainset_Raw_Y = mat_file['train_Y']
        trainset_Raw = np.concatenate((trainset_Raw_X, trainset_Raw_X_PP, trainset_Raw_Y), axis=1)
        
        testset_Raw_X = mat_file['test_X']
        testset_Raw_X_PP = mat_file['test_PP_spl']
        testset_Raw_Y = mat_file['test_Y']
        testset_Raw = np.concatenate((testset_Raw_X, testset_Raw_X_PP, testset_Raw_Y), axis=1)
    
#%% 
    dataset = np.concatenate((trainset_Raw,testset_Raw), axis=0)
    mean_Out = np.mean(dataset[:,-1])
    if norm == 1:
        
        lb_Out=np.min(dataset[:,-1])
        ub_Out=np.max(dataset[:,-1])    
        trainset = copy.deepcopy(trainset_Raw)
        trainset[:,-1] = (((trainset_Raw[:,-1] - lb_Out) * (1 - 0)) / (ub_Out - lb_Out)) + 0
        testset = copy.deepcopy(testset_Raw)
        testset[:,-1] = (((testset_Raw[:,-1] - lb_Out) * (1 - 0)) / (ub_Out - lb_Out)) + 0
    else:
        trainset = copy.deepcopy(trainset_Raw)
        testset = copy.deepcopy(testset_Raw)
    
    #%%
    
    x_test = torch.Tensor(testset[:,:c.D_in])
    y_test = torch.Tensor(testset[:,c.D_in:])
    x_train = torch.Tensor(trainset[:,:c.D_in])
    y_train = torch.Tensor(trainset[:,c.D_in:])
    x_train = torch.Tensor(x_train).to(c.device)
    y_train = torch.Tensor(y_train).to(c.device)
    x_test= torch.Tensor(x_test).to(c.device)
    y_test = torch.Tensor(y_test).to(c.device)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test),
        batch_size=c.batch_size, shuffle=True, drop_last=True)
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=c.batch_size, shuffle=True, drop_last=True)
    
    cd = {
        'network_size' : c.Num_layers,
        'dropout': c.dropout,
        'hidden_layer_size':c.Hidden_layer_size
    }
    ##%% Initalize model and optimizer
    model = Fully_connected(c.D_in,c.D_out,cd)
    print(c.D_in,c.D_out,c.Num_layers,c.Hidden_layer_size)
    model.to(c.device)
    optimizer = torch.optim.Adam(model.parameters(),c.lr,weight_decay=c.weight_decay)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=1, patience=100, threshold=0.00001, threshold_mode='rel', cooldown=150, min_lr=1e-6, eps=1e-08, verbose=False)
    ##Actual Training.........
    train_step = make_train_step(model,optimizer)
    training_loss = []
    test_loss =[]
    ConvHistLoss=torch.rand((c.epochs,1))
    for epoch in range(c.epochs):
        batch_losses = []
        for x_batch, y_batch in train_loader:
            loss = train_step(x_batch, y_batch)
            batch_losses.append(loss)
        training_loss.append(np.mean(batch_losses))
        batch_losses =[]
        for x_batch,y_batch in test_loader:
            loss = train_step(x_batch, y_batch,test=True)
            batch_losses.append(loss)
        test_loss.append(np.mean(batch_losses))
        ConvHistLoss[epoch,0]=training_loss[-1]
        print('epoch',epoch,'training loss',training_loss[-1], 'test loss',test_loss[-1])
    
    #%% Saving data
    U = model(x_test); #pos_vec1 = model_input;
    tempnp=U.cpu()
    temp2=tempnp.detach().numpy()
    
    if norm ==1:
        Y_pred = (temp2*(ub_Out -lb_Out)) + lb_Out 
    else:
        Y_pred = copy.deepcopy(temp2)
        
    # with open('Testset_Pred', 'wb') as f:
    #     pickle.dump(Y_pred, f)
        
    rae=copy.deepcopy(Y_pred);re=copy.deepcopy(Y_pred)
    rae[:,0] = np.absolute(Y_pred[:,0]-testset_Raw[:,-1])/mean_Out *100
    re[:,0] = (Y_pred[:,0]-testset_Raw[:,-1])/mean_Out *100
    mean_rae = np.mean(rae)
    # mdic = {"Y_pred": Y_pred, "re": re, "rae": rae, "mean_rae": mean_rae, "testset_Raw": testset_Raw}
    # scipy.io.savemat("matlab_matrix.mat", mdic)
    
    lala = ConvHistLoss.cpu()
    ConvHist = lala.detach().numpy()
    # with open('ConvHist', 'wb') as f:
    #     pickle.dump(ConvHist, f)
    
    torch.save(model.state_dict(), 'output/trained_model.pt')
    
    return Y_pred, rae, re, mean_rae, ConvHist, ub_Out, lb_Out
