%% Sample Call Partial Physics
% Creates a dataset to train the ML models
% -------------------------------------------
% Load the experimentally calculated data of the UAV.
% Define the parameters for the partial physics model under the structure param.
% Call the partial physics model .
% Save the output dataset as data.mat.
%--------------------------------------------

%% Loads the experimental data as training and testing data
clc;    clear all;

load('Experimental_Data.mat') 

%% Set the parameters for the partial physics model

param.phi = [45, 45, 45, 45];
param.freq = [175, 175, 175, 175];
param.t_end = 0; param.T = 1; param.samp_freq = 1000; param.c = 343; param.P_ref = 2.000000000000000e-05; param.n=4;
param.mono_loc = [0.176776695296637,-0.176776695296637,-0.176776695296637,0.176776695296637;0.176776695296637,0.176776695296637,-0.176776695296637,-0.176776695296637;0,0,0,0];

%% Define the U values for seperate number of monopole cales

U1=[[1];[0.5];[1.5];[-1];[-0.5]];
U2=[[1,1]; [0.5,1]; [0.5, 0.5]; [-0.5, 0.5]; [-0.5, 1]];
U3=[[1,1,1]; [0.5,1,1]; [1,0.5,1]; [1,1,0.5]; [0.5, 0.5,0.5]];
U4=[[1,1,1,1]; [0.5,1,1,1]; [1,0.5,1,1]; [1,1,0.5,1]; [1,1,1,0.5]];
U5=[[1,1,1,1,1]; [0.5,1,1,1,1]; [1,0.5,1,1,1]; [1,1,0.5,1,1]; [1,1,1,0.5,1]];

%Selecting number of monopoles as 4
U = U4;

%% Generating partila physics values for the given setting

for i = 1:5
    
    U_now = repmat(U(i,:),815,1);
    train_spl(:,i) = PartialPhysics(U_now, train_X, param);
    U_now = repmat(U(i,:),913,1);
    test_spl(:,i) = PartialPhysics(U_now, test_X, param);
end
train_PP_spl = train_spl;
test_PP_spl = test_spl;

%% Saving dataset

save('data.mat','train_X','train_Y','train_PP_spl','test_X','test_Y','test_PP_spl')