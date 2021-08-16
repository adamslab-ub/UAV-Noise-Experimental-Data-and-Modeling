import torch
import torch.nn
import config1 as c
import numpy as np
#from Call_Partial import Custom_Loss
#from Call_Partial_Numeric import Partial_Phys1
#from CustomLayers import *




# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.sparse_(m.weight,sparsity=0.9)
        m.bias.data.fill_(1)
class Fully_connected(torch.nn.Module):
    def __init__(self, D_in, D_out,config):
        super(Fully_connected, self).__init__()
        self.layers = torch.nn.ModuleList()
        H = config['hidden_layer_size']
        #self.drop = torch.nn.ModuleList()
        self.norm = torch.nn.BatchNorm1d(D_in)
        self.linear_in = torch.nn.Linear(D_in, H)
        self.dropoutp = config['dropout']

        for i in range(c.Num_layers):
            self.layers.append(torch.nn.Linear(H,H))
        self.drop = torch.nn.Dropout(p=self.dropoutp)
        self.linear_out = torch.nn.Linear(H, D_out)
        #self.nl1 = torch.nn.ReLU()
        #self.nl1 = torch.nn.LeakyReLU(negative_slope=1)
        self.nl1 = torch.nn.ReLU()

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        out = self.linear_in(self.norm(x))
        for i in range(len(self.layers)):
            net = self.layers[i]
            out = self.nl1(self.drop(net(out)))
        out = self.linear_out(out)
        #out = torch.nn.ReLU(out)# + 0.5

        # p_uav1 = torch.zeros(out.shape[0],1,dtype=torch.cfloat).to(c.device)
        # for n in range (0,4):
        #     r = torch.sqrt( torch.pow( x[:,0] - c.mono_loc[0,n], 2) + torch.pow( x[:,1] - c.mono_loc[1,n], 2) + torch.pow( x[:,2] - c.mono_loc[2,n], 2))
        #     p_uav1[:,0] = p_uav1[:,0] + (out[:,n]* torch.exp(c.comp_1i*( - c.kappa[n]*r + c.phi[0,n])))/r
    
        # temp = c.T0*p_uav1/c.P_ref
        # re= torch.sqrt( torch.pow(temp.real,2) + torch.pow(temp.imag,2))
        # spl_mic_main1 = 20* torch.log10( torch.abs(re))
        # return Normalize(spl_mic_main1)
        
        
        
        #return Normalize(out)
        return out
        #return self.nl2(out)
        #return torch.divide(torch.sin(10*3.1415927410125732*(out)), 2*(out)) +  torch.pow((out) -1, 4)
#def l2_loss(input, target, model_input):    
def l2_loss(input, target):    
#     #x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)

#     # input = torch.cuda.FloatTensor([[1,1,1,1]]);
#     # model_input = torch.cuda.FloatTensor([[0.000436737,	0.469495,	0.896088]]);
#     # target=torch.cuda.FloatTensor([[4]]); 
#     #print(input)
#     ub = torch.cuda.FloatTensor([[-0.0194488,1.56121,0.627335]])
#     lb = torch.cuda.FloatTensor([[-1.61423,-1.38,-1.6437]])
    
#     #print(input.requires_grad)
#     #U = input; #pos_vec1 = model_input;
#     #print(U[0,:])
#     pos_vec1 = torch.cuda.FloatTensor(model_input.shape)
#     for i in range(0,3):
#         #pos_vec1[:,i]= ((model_input[:,i] * 4) - 2)
#         pos_vec1[:,i]= ((model_input[:,i] * (ub[0,i] - lb[0,i])) + lb[0,i])
#         #pos_vec1[i]= ((model_input[i] * (ub[0,i] - lb[0,i])) + lb[0,i])
    
#     # print(model_input.shape)
#     pi = torch.acos(torch.zeros(1)).item() * 2
#     # U=torch.tensor([[1,1,1,1]]); pos_vec1=torch.tensor([[-1,-1,-1]]);
#     #print(U)
#     N=torch.cuda.FloatTensor([[4]]); 

#     mono_loc=torch.cuda.FloatTensor([[0.642718009121920,	-0.769149253898193,	-1.99925545455873,	1.99747800231622], 
#                            [-0.632032920724389,	-1.98522154843943,	-1.99955394174514,	-1.99525104388291],
#                            [-0.531431436623549,	-1.99485413030846,	-1.99924280318578,	1.99496038586142]])
#     freq=torch.cuda.FloatTensor([[94.0512457240110,	0.000776299875521865,	134.229782863088,	0.000888841271693431]])
    
#     #rho = torch.cuda.FloatTensor([[1.2]]); c_uav=torch.cuda.FloatTensor([[343]]); a_uav=torch.cuda.FloatTensor([[0.003]]); T0=torch.cuda.FloatTensor([[1]]);
#     phi = torch.cuda.FloatTensor([[45.0016347524273,	0.997894925987488,	45.3771593274238,	0.000439932865296745]]); c_uav =torch.cuda.FloatTensor([[343]]); T0=torch.cuda.FloatTensor([[1]]);
#     P_ref = torch.cuda.FloatTensor([[20e-6]])
#     comp_1i = torch.tensor([[0.0 + 1j]]).to(c.device)
#     xxx = torch.randn(2,2, dtype=torch.cfloat)
#     time = torch.cuda.FloatTensor([[0]])        
#     ang_freq = 2*pi*freq[0,:]
#     #freq_weights = freq[1,:]
    
#     P = torch.tensor([[0.0 + 0.0j]]).to(c.device)
#     p_uav = torch.tensor([[0.0 + 0.0j]]).to(c.device)
#     p_o = torch.tensor([[0.0 + 0.0j]]).to(c.device)
    
#     t = 0
    
#     spl_mic_main = torch.cuda.FloatTensor(target.shape)
#     # print(target.shape)
#     # print(spl_mic_main.shape)
#     # for i in range(0,U.shape[0]):
#     for i in range(0,input.shape[0]):
#         #U0 = U[i,:]
#         pos_vec = pos_vec1[i,:]
#         #pos_vec = pos_vec1[:]
#         # print("i",pos_vec)
#         # print("ii",model_input[i,:])
#         # print("iii",target[i,:])
        
#         for n in range (0,4):
#             #r = torch.sqrt( torch.pow( pos_vec[0,0] - mono_loc[0,n], 2) + torch.pow( pos_vec[0,1] - mono_loc[1,n], 2) + torch.pow( pos_vec[0,2] - mono_loc[2,n], 2))
#             r = torch.sqrt( torch.pow( pos_vec[0] - mono_loc[0,n], 2) + torch.pow( pos_vec[1] - mono_loc[1,n], 2) + torch.pow( pos_vec[2] - mono_loc[2,n], 2))
#             kappa = ang_freq[n]/c_uav
#             p_uav[:,t] = p_uav[:,t] + (input[i,n]* torch.exp(comp_1i*(ang_freq[n]*time[t] - kappa*r + phi[0,n])))/r
# 			#r = torch.sqrt( torch.pow( pos_vec[0,0] - mono_loc[0,n], 2) + torch.pow( pos_vec[0,1] - mono_loc[1,n], 2) + torch.pow( pos_vec[0,2] - mono_loc[2,n], 2))
#             #for o in range(0,ang_freq.shape[0]):
#             #    kappa = ang_freq[o]/c_uav
#             #    p_o[:,t] = p_o[:,t] + freq_weights[o]*comp_1i*rho*c_uav*kappa*(torch.pow(a_uav,2))*(input[i,n]* torch.exp(comp_1i*(ang_freq[o]*time[t] - kappa*r)))/r
#             #    #p_o[:,t] = p_o[:,t] + freq_weights[o]*comp_1i*rho*c_uav*kappa*(torch.pow(a_uav,2))*(U0[0,n]* torch.exp(comp_1i*(ang_freq[o]*time[t] - kappa*r)))/r
#             #p_uav[:,t] = p_uav[:,t] + p_o[:,t]
#         P[:,t] = P[:,t] + p_uav[:,t]
        
#         temp = T0*P/P_ref
#         r= torch.sqrt( torch.pow(temp.real,2) + torch.pow(temp.imag,2))
#         spl_mic_main[i] = 20* torch.log10( torch.abs(r))
#         # print(spl_mic_main)
#         # spl_mic = (20*torch.log10(T0*P/P_ref))
#         # spl_mic = (20*(T0*P/P_ref)/1000);
#         # spl_mic_main[i] = spl_mic.real
#     # print(spl_mic.is_cuda)
#     #print("Raw",spl_mic_main)
#     spl_mic_main= (spl_mic_main - 75.75212099867892)/(83.27268669594696 - 75.75212099867892)
#     #print("X",pos_vec1[0,:])
#     # print("U",input)
#     #print("Normalized",spl_mic_main[0])
#     # print("XYZ",model_input)
#     # print("target",target)
    
     #spl = Custom_Loss(input, target, model_input)
     #######spl = Partial_Phys1(input)
     #print(spl.shape)
     #print(target)
     loss = torch.nn.MSELoss()
     #loss = torch.nn.L1Loss()
     #print(input)
     #return loss(spl,target)
     return loss(input,target)
def make_train_step(model,optimizer,scheduler=None):
    # Builds function that performs a step in the train loop
    def train_step(x, y,test=False):
        a=model
        if not test:
            
            
            yhat = a(x)
            #print(yhat.requires_grad)
            #loss = l2_loss(yhat, y, x)
            loss = l2_loss(yhat, y)
            #print(yhat)
            optimizer.zero_grad()
            loss.backward()
            #print(model.layers[0].weight.grad[0,:])
            # torch.autograd.gradcheck()
            optimizer.step()
        else:
            a = model.eval()
            with torch.no_grad():
                yhat = a(x)
                #loss = l2_loss(yhat, y, x)
                loss = l2_loss(yhat, y)
            if scheduler:
                scheduler.step(loss)
        return loss.item()
    # Returns the function that will be called inside the train loop
    return train_step
