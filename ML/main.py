#%% Main File to create machne learning models
#
"""

@author: Rayhaan Iqbal

# Parameters:
# model ==> 1: A Pure data driven model; 2: Sequential Hybrid Physics Infused Machine Learning model
# norm  ==> 0: Do not normalize target values; 1: Normalize the target values
# data  ==> training data from the UAV noise problem saved as a .mat file
#
# Sample of application of the trained model shown in second half
"""
# Parameters:
# model ==> 1: A Pure data driven model; 2: Sequential Hybrid Physics Infused Machine Learning model
# norm  ==> 0: Do not normalize target values; 1: Normalize the target values
# data  ==> training data from the UAV noise problem saved as a .mat file
#
# Sample of application of the trained model shown in second half

#%% Building the models
import numpy as np
import torch.utils.data
from functools import partial
import os
from Machine_Learning import *

model=1; norm=1

from scipy.io import loadmat
mat_file = loadmat('data.mat')

Y_pred, rae, re, mean_rae, ConvHist, ub_Out, lb_Out = build_model(model, mat_file)

#%% Importing trained model

if model ==1:
    import config1 as c
elif model == 2:
    import config1 as c

cd = {
    'network_size' : c.Num_layers,
    'dropout': c.dropout,
    'hidden_layer_size':c.Hidden_layer_size
}
#Initalize model and optimizer
model = Fully_connected(c.D_in,c.D_out,cd)
model.to(c.device)
model.load_state_dict(torch.load('output/trained_model.pt'))
model.eval()

# sample testing location with coordinates (1,1,1)
sample_data=torch.cuda.FloatTensor([[1,1,1]])
out = model(sample_data)
out=out.cpu()
out=out.detach().numpy()
if norm ==1:
    Y_pred = (out*(ub_Out -lb_Out)) + lb_Out 