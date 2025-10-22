# %% [markdown]
# ## Import modules

# %%
# math
import numpy as np
# plots stuff
import matplotlib as mlp
mlp.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import shutil
from IPython.display import clear_output
import time
from time import sleep
import sys
args=sys.argv[1:]
#  import NN training hyperparameters from outside
#  lr decayrate weightdecay gamma alpha nn1 nn2 
_lr_=float(args[0]);_decayrate_=float(args[1]);_weightdecay_=float(args[2]);_gamma_=float(args[3])

_alpha_=float(args[4]);_nn1_=int(args[5]); _nn2_=int(args[6]); _iterid_=int(args[7]);_logbase_=float(args[8]);_Loadingset_=args[9]
print(_Loadingset_)
Loading_bool = _Loadingset_ == 'True' 
print(f'Loading_bool{Loading_bool}')
# mlcolvar
import mlcolvar
from mlcolvar.data import DictDataset,DictModule
from mlcolvar.cvs import DeepTDA
from mlcolvar.utils.fes import compute_fes
from mlcolvar.utils.io import create_dataset_from_files,load_dataframe
from mlcolvar.utils.plot import plot_isolines_2D, muller_brown_potential, muller_brown_mfep, paletteFessa, paletteCortina

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from typing import List


# torch
import torch
from torch import nn
from torch.utils.data import random_split

# torch.manual_seed(42)
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print ('Torch device: ', device)
torch.set_default_device(device)

# %% [markdown]
# # Functions and Classes
# 

# %% [markdown]
# ### Custom activation
# 
# $$\frac{1}{1 + exp[-px]}$$
# 
# Linear activation is not enough for the last layer.
# If we add an activation like a sharper sigmoid it helps to collapse the q(x) to constant values in the metastable states.
# 
# This way we can also ensure that it never goes below zero if we want to use it as a CV.
# 
# TODO: Check if necessary!

# %%
class CustomActivation(nn.Module):
    def __init__(self, p):
        super(CustomActivation, self).__init__()
        self.p = p

    def forward(self, x):
        return 1 / (1 + torch.exp(-self.p*(x)))

# %% [markdown]
# ### Committor guess model: $q_\theta(x)$
# 
# Open questions on deltaF and beta, we use them only in muller!!
# - $\Delta F$ can make things faster in the earlier stage, we added a keyword to pass it or not
# 
# It is helpful also in Alanine actually but of course it complicates things a little bit because you need more informations on the system.

# %%
class NN_committor_guess(nn.Module):
    def __init__(self, 
                 nnLayers : list, 
                 p : float = 3,
                 preprocessing : nn.Module = None):
        super(NN_committor_guess, self).__init__() 
        
        # initalize attributes
        self.preprocessing = preprocessing

        # initialize the NN
        modules = []
        for i in range(len(nnLayers)-1):
            print(nnLayers[i], ' --> ', nnLayers[i+1], end=' ')
            if (i<len(nnLayers)-2): 
                modules.append(nn.Linear(nnLayers[i], nnLayers[i+1])) # connection line
               # modules.append(nn.Dropout(0.05)) # dropout layer
                modules.append(nn.Tanh()) # non linearity = activation function of nth nodes
                print('(Tanh)', end = '')
            else:  # last layer only
                modules.append(nn.Linear(nnLayers[i], nnLayers[i+1]))
                modules.append(CustomActivation(p))
                print('')
            print()

        # This actually makes the NN a pytorch object
        self.nn = nn.Sequential(*modules) # the star unpacks the elements in the array

        # what the actual model does
    def forward(self, x : torch.Tensor):
        if self.preprocessing is not None:
            x = self.preprocessing(x)
        q = self.nn(x)
        return q

# %%
class NN_committor_zeta(nn.Module):
    def __init__(self, 
                 nnLayers : list, 
                 p : float = 5,
                 preprocessing : nn.Module = None):
        super(NN_committor_zeta, self).__init__() 
        
        # initalize attributes
        self.preprocessing = preprocessing

        # initialize the NN
        modules = []
        for i in range(len(nnLayers)-1):
            print(nnLayers[i], ' --> ', nnLayers[i+1], end=' ')
            if (i<len(nnLayers)-2): 
                modules.append(nn.Linear(nnLayers[i], nnLayers[i+1])) # connection line
                modules.append(nn.Tanh()) # non linearity = activation function of nth nodes
                print('(Tanh)', end = '')
            else:  # last layer only
                modules.append(nn.Linear(nnLayers[i], nnLayers[i+1]))
                print('')
            print()

        # This actually makes the NN a pytorch object
        self.nn = nn.Sequential(*modules) # the star unpacks the elements in the array

        # what the actual model does
    def forward(self, x : torch.Tensor):
        if self.preprocessing is not None:
            x = self.preprocessing(x)
        q = self.nn(x)
        return q
      

# %% [markdown]
# ### Loss Function
# 
# Open questions:
# - Check if grad needs to be divided due to reduced coordinates (maybe it was only for PDE term)

# %%
def loss_function(x : torch.Tensor, 
                  q : torch.Tensor,
                  dx :  torch.Tensor,
                  labels: torch.Tensor, 
                  w: torch.Tensor,
                  mass: torch.Tensor,
                  alpha : float,
                  cell_size: float = None,
                  gamma : float = 10000,
                  delta_f: float = 0,
                  create_graph : bool = True):
    """Compute variational loss for committor optimization with boundary conditions

    Parameters
    ----------
    x : torch.Tensor
        Input of the NN
    q : torch.Tensor
        Committor quess q(x), it is the output of NN
    labels : torch.Tensor
        Labels for states, A and B states for boundary conditions
    w : torch.Tensor
        Reweighing factors. This depends on the simualtion in which the data were collected.
        It is standard reweighing: exp[-beta*V(x)]
    mass : torch.Tensor
        List of masses of all the atoms we are using, for each atom we need to repeat three times for x,y,z
    alpha : float
        Hyperparamer that scales the boundary conditions contribution to loss, i.e. alpha*(loss_bound_A + loss_bound_B)
    cell_size : float
        CUBIC cell size length, used to scale the positions from reduce coordinates to real coordinates, default None 
    gamma : float
        Hyperparamer that scales the whole loss to avoid too small numbers, i.e. gamma*(loss_var + loss_bound) 
        By default 10000
    delta_f : float
        Delta free energy between A (label 0) and B (label 1), units is kBT, by default 0. 
        State B is supposed to be higher in energy.
    create_graph : bool
        Make loss backwardable, deactivate for validation to save memory, default True
    """

    # Create masks to access different states data
    mask_A = torch.nonzero(labels.squeeze() == 0, as_tuple=True) 
    mask_B = torch.nonzero(labels.squeeze() == 1, as_tuple=True) 
    
    # Update weights of basin B using the information on the delta_f
    delta_f = torch.Tensor([delta_f])
    if delta_f < 0: # B higher in energy --> A-B < 0
        w[mask_B] = w[mask_B] * torch.exp(delta_f.to(device))
    elif delta_f > 0: # A higher in energy --> A-B > 0
        w[mask_A] = w[mask_A] * torch.exp(delta_f.to(device))
    ###### VARIATIONAL PRINICIPLE LOSS ######
    # Each loss contribution is scaled by the number of samples
    w[mask_B]=0
    w[mask_A]=0
    # We need the gradient of q(x)
    grad_outputs = torch.ones_like(q)
    grad = torch.autograd.grad(q, x, grad_outputs=grad_outputs, retain_graph=True, create_graph=create_graph)[0]
    # print (grad.shape)  # 1800 5
    grad_expanded = grad.unsqueeze(1) 
    # print (grad_expanded.shape) # 1800 1 5
    gradall = grad_expanded*dx
    # print (gradall.shape) # 1800 8748 5
    gradall = torch.sum(gradall,axis=2)
    # print (gradall.shape) # 1800 8748
    # # TODO this fixes cell size issue
    # if cell_size is not None:
    #     grad = grad / cell_size

    # we sanitize the shapes of mass and weights tensors
    # mass should have size [1, n_atoms*spatial_dims]
    mass = mass.unsqueeze(0)
    # weights should have size [n_batch, 1]
    w = w.unsqueeze(-1)
    grad_square = torch.sum((torch.pow(gradall, 2)*(1/mass)), axis=1, keepdim=True) * w
    # print (grad_square.shape) # 1800 1
    # variational contribution to loss: we sum over the batch
    loss_var = torch.mean(grad_square)

    # boundary conditions
    q_A = q[mask_A]
    q_B = q[mask_B]
    loss_A = torch.mean( torch.pow(q_A, 2))
    loss_B = torch.mean( torch.pow((q_B - 1), 2))

    # b=_logbase_
    loss = torch.log(loss_var) + gamma * alpha*(loss_A + loss_B)
    # cont_ana2=torch.isnan(loss).any()
    # print(f'loss contain nan :{cont_ana2}')

    return loss, [loss_var.detach(), alpha*loss_A.detach(), alpha*loss_B.detach()]

# %% [markdown]
# ### printrealloss

# %%
def printrealloss(x : torch.Tensor, 
                  q : torch.Tensor,
                  dx :  torch.Tensor,
                  labels: torch.Tensor, 
                  w: torch.Tensor,
                  mass: torch.Tensor,
                  alpha : float,
                  cell_size: float = None,
                  gamma : float = 10000,
                  delta_f: float = 0,
                  create_graph : bool = True,
                  iterr: float = 1
                  ):
    """Compute variational loss for committor optimization with boundary conditions

    Parameters
    ----------
    x : torch.Tensor
        Input of the NN
    q : torch.Tensor
        Committor quess q(x), it is the output of NN
    labels : torch.Tensor
        Labels for states, A and B states for boundary conditions
    w : torch.Tensor
        Reweighing factors. This depends on the simualtion in which the data were collected.
        It is standard reweighing: exp[-beta*V(x)]
    mass : torch.Tensor
        List of masses of all the atoms we are using, for each atom we need to repeat three times for x,y,z
    alpha : float
        Hyperparamer that scales the boundary conditions contribution to loss, i.e. alpha*(loss_bound_A + loss_bound_B)
    cell_size : float
        CUBIC cell size length, used to scale the positions from reduce coordinates to real coordinates, default None 
    gamma : float
        Hyperparamer that scales the whole loss to avoid too small numbers, i.e. gamma*(loss_var + loss_bound) 
        By default 10000
    delta_f : float
        Delta free energy between A (label 0) and B (label 1), units is kBT, by default 0. 
        State B is supposed to be higher in energy.
    create_graph : bool
        Make loss backwardable, deactivate for validation to save memory, default True
    """


    ###### VARIATIONAL PRINICIPLE LOSS ######
    # Each loss contribution is scaled by the number of samples
    
    # We need the gradient of q(x)
    grad_outputs = torch.ones_like(q)
    grad = torch.autograd.grad(q, x, grad_outputs=grad_outputs, retain_graph=True, create_graph=create_graph)[0]
    # print (grad.shape)  # 1800 5

    grad_expanded = grad.unsqueeze(1) 
    # print (grad_expanded.shape) # 1800 1 5
    # print (dx.shape)
    gradall = grad_expanded*dx
    # print (gradall.shape) # 1800 8748 5

    gradall = torch.sum(gradall,axis=2)
    # print (gradall.shape) # 1800 8748

    # # TODO this fixes cell size issue
    # if cell_size is not None:
    #     grad = grad / cell_size

    # we sanitize the shapes of mass and weights tensors
    # mass should have size [1, n_atoms*spatial_dims]
    mass = mass.unsqueeze(0)

    # we get the square of grad(q) and we don't multiply by the weight because we just
    # want to see the sample distribution relationship with grad(q)
    
    grad_square_true = torch.sum((torch.pow(gradall, 2)*(1/mass)), axis=1, keepdim=True)
    # * w
    condition=(q.cpu().detach().numpy()[:]<0.6)&(q.cpu().detach().numpy()[:]>0.4)
    condition=condition.reshape((-1,))
    TSEtemp=x.cpu().detach().numpy()[:,1]
    TSEtemp=(TSEtemp[condition])*4.149466290000000299e+02 +1.325499899999999975e+01 

    condition2=(q.cpu().detach().numpy()[:]<0.7)&(q.cpu().detach().numpy()[:]>0.3)
    condition2=condition2.reshape((-1,))
    TSEtemp2=x.cpu().detach().numpy()[:,1]
    TSEtemp2=(TSEtemp2[condition2])*4.149466290000000299e+02 +1.325499899999999975e+01  
    # print('##########333')
    # print(q.cpu().detach().numpy(),grad_square_true.cpu().detach().numpy())

    # print the TSE location and the TSE distribution things

    print(f'TSEnum: {len(TSEtemp)}, TSEnum lr: {len(TSEtemp2)}')
    print(f'TSEave: {np.mean(TSEtemp)}, TSEave lr: {np.mean(TSEtemp2)}')
    plt.figure()
    plt.scatter(q.cpu().detach().numpy(),grad_square_true.cpu().detach().numpy(),alpha=0.5,s=1.2)
    plt.title(f'TSEnum: {len(TSEtemp)}, TSEave: {np.mean(TSEtemp)}')
    plt.savefig(f'bias{int(iterr)}_nobeta.png')
    plt.close()
    return len(TSEtemp)/len(condition),len(TSEtemp2)/len(condition2)
    # return q.cpu().detach().numpy(),grad_square_true.cpu().detach().numpy()

# %% [markdown]
# ### Compute weights from biased simulations

# %%
def compute_weights_mod(dataframe, dataset, beta : float, mixing : bool, mixing_csi : float,factor:float):
    # Check if we have the bias column and sanitize it
    if 'bias' in dataframe.columns:
        dataframe = dataframe.fillna({'bias': 0})
    else:
        dataframe['bias'] = 0

    # compute weights
    dataframe['weights'] = np.exp(beta * dataframe['bias'])

    # group data from same iteration under the same label, keep 0 and 1 safe because unbiased!
    dataframe.loc[np.logical_and(dataframe['labels'] % 2 == 1 , dataframe['labels'] > 1), 'labels'] = dataframe.loc[np.logical_and(dataframe['labels'] % 2 == 1 , dataframe['labels'] > 1), 'labels'] - 1
    dataframe.loc[dataframe['labels'] > 1, 'labels'] = dataframe.loc[dataframe['labels'] > 1, 'labels'] / 2 + 1

    # get the reweight averages
    for i in np.unique(dataframe['labels'].values):
        # compute average of exp(beta*V) on this simualtions
        # delete +0.0000001
        coeff = 1 / (np.mean(dataframe.loc[dataframe['labels'] == i, 'weights'].values))/factor
        # coeff = 1 / (np.mean(dataframe.loc[dataframe['labels'] == i, 'weights'].values) +0.0000001)

        # we apply weight mixing between iterations: more weight to last iterations   
        if mixing:
            max_iter = np.max(np.unique(dataframe['labels'].values))
            if i>1:
                coeff = coeff * ( mixing_csi**(max_iter - i) * (1 - mixing_csi))
            else:
                coeff =1# coeff * (mixing_csi**(max_iter - 1))
        print (coeff)    
        # update the weights
        dataframe.loc[dataframe['labels'] == i, 'weights'] = coeff * dataframe.loc[dataframe['labels'] == i, 'weights']
    
    # update labels add weights to torch dataset
    dataset['labels'] = torch.Tensor(dataframe['labels'].values)
    dataset['weights'] = torch.Tensor(dataframe['weights'].values)
    
    return dataframe, dataset

# %%
def compute_weights_v2(dataframe, dataset, beta : float, mixing : bool, mixing_csi : float):
    # Check if we have the bias column and sanitize it
    if 'bias' in dataframe.columns:
        dataframe = dataframe.fillna({'bias': 0})
    else:
        dataframe['bias'] = 0

    # compute weights
    dataframe['weights'] = np.exp(beta * dataframe['bias'])

    # group data from same iteration under the same label, keep 0 and 1 safe because unbiased!
    dataframe.loc[np.logical_and(dataframe['labels'] % 4 == 1 , dataframe['labels'] > 1), 'labels'] = dataframe.loc[np.logical_and(dataframe['labels'] % 4 == 1 , dataframe['labels'] > 1), 'labels'] - 1
    dataframe.loc[np.logical_and(dataframe['labels'] % 4 == 2 , dataframe['labels'] > 1), 'labels'] = dataframe.loc[np.logical_and(dataframe['labels'] % 4 == 2 , dataframe['labels'] > 1), 'labels'] +2
    dataframe.loc[np.logical_and(dataframe['labels'] % 4 == 3 , dataframe['labels'] > 1), 'labels'] = dataframe.loc[np.logical_and(dataframe['labels'] % 4 == 3 , dataframe['labels'] > 1), 'labels'] +1
    dataframe.loc[dataframe['labels'] > 1, 'labels'] = dataframe.loc[dataframe['labels'] > 1, 'labels'] / 4 + 1

    # get the reweight averages
    for i in np.unique(dataframe['labels'].values):
        # compute average of exp(beta*V) on this simualtions
        coeff = 1 / (np.mean(dataframe.loc[dataframe['labels'] == i, 'weights'].values)+0.0000001)

        # we apply weight mixing between iterations: more weight to last iterations   
        if mixing:
            max_iter = np.max(np.unique(dataframe['labels'].values))
            if i>1:
                coeff = coeff * ( mixing_csi**(max_iter - i) * (1 - mixing_csi))
            else:
                coeff = coeff * (mixing_csi**(max_iter - 1))
        print ('coeff')  ;        print (coeff)    
  
        # update the weights
        dataframe.loc[dataframe['labels'] == i, 'weights'] = coeff * dataframe.loc[dataframe['labels'] == i, 'weights']
    
    # update labels add weights to torch dataset
    dataset['labels'] = torch.Tensor(dataframe['labels'].values)
    dataset['weights'] = torch.Tensor(dataframe['weights'].values)
    
    return dataframe, dataset

# %%
def compute_weights_v3(dataframe, dataset, beta : float, mixing : bool, mixing_csi : float):
    # Check if we have the bias column and sanitize it
    if 'bias' in dataframe.columns:
        dataframe = dataframe.fillna({'bias': 0})
    else:
        dataframe['bias'] = 0

    # compute weights
    dataframe['weights'] = np.exp(beta * dataframe['bias'])

    # group data from same iteration under the same label, keep 0 and 1 safe because unbiased!
    dataframe.loc[np.logical_and(dataframe['labels'] % 4 == 1 , dataframe['labels'] > 1), 'labels'] = dataframe.loc[np.logical_and(dataframe['labels'] % 4 == 1 , dataframe['labels'] > 1), 'labels'] - 1
    dataframe.loc[np.logical_and(dataframe['labels'] % 4 == 2 , dataframe['labels'] > 1), 'labels'] = dataframe.loc[np.logical_and(dataframe['labels'] % 4 == 2 , dataframe['labels'] > 1), 'labels'] +2
    dataframe.loc[np.logical_and(dataframe['labels'] % 4 == 3 , dataframe['labels'] > 1), 'labels'] = dataframe.loc[np.logical_and(dataframe['labels'] % 4 == 3 , dataframe['labels'] > 1), 'labels'] +1
    dataframe.loc[dataframe['labels'] > 1, 'labels'] = dataframe.loc[dataframe['labels'] > 1, 'labels'] / 4 + 1

    # get the reweight averages
    for i in np.unique(dataframe['labels'].values):
        # compute average of exp(beta*V) on this simualtions
        #### note that i delete the 1e-8 term here
        coeff = 1 / (np.mean(dataframe.loc[dataframe['labels'] == i, 'weights'].values))

        # we apply weight mixing between iterations: more weight to last iterations   
        if mixing:
            max_iter = np.max(np.unique(dataframe['labels'].values))
            if i>1:
                coeff = coeff * ( mixing_csi**(max_iter - i) * (1 - mixing_csi))
            else:
                coeff = coeff * (mixing_csi**(max_iter - 1))
        print ('coeff')  ;        print (coeff)    
  
        # update the weights
        dataframe.loc[dataframe['labels'] == i, 'weights'] = coeff * dataframe.loc[dataframe['labels'] == i, 'weights']
    
    # update labels add weights to torch dataset
    dataset['labels'] = torch.Tensor(dataframe['labels'].values)
    dataset['weights'] = torch.Tensor(dataframe['weights'].values)
    
    return dataframe, dataset


def compute_committor_weights(dataset, 
                              bias: torch.Tensor, 
                              data_groups: List[int], 
                              beta: float):
    """Utils to update a DictDataset object with the appropriate weights and labels for the training set for the learning of committor function.

    Parameters
    ----------
    dataset : 
        Labeled dataset containig data from different simulations, the labels must identify each of them. 
        For example, it can be created using `mlcolvar.utils.io.create_dataset_from_files(filenames=[file1, ..., fileN], ... , create_labels=True)`
    bias : torch.Tensor
        Bias values for the data in the dataset, usually it should be the committor-based bias
    data_groups : List[int]
        Indices specyfing the iteration each labeled data group belongs to. 
        Unbiased simulations in A and B used for the boundary conditions must have indices 0 and 1.
    beta : float
        Inverse temperature in the right energy units

    Returns
    -------
        Updated dataset with weights and updated labels
    """

    if bias.isnan().any():
        raise(ValueError('Found Nan(s) in bias tensor. Check before proceeding! If no bias was applied replace Nan with zero!'))

    # TODO sign if not from committor bias
    weights = torch.exp(beta * bias)
    new_labels = torch.zeros_like(dataset['labels'])

    data_groups = torch.Tensor(data_groups)

    # correct data labels according to iteration
    for j,index in enumerate(data_groups):
        new_labels[torch.nonzero(dataset['labels'] == j, as_tuple=True)] = index

    for i in np.unique(data_groups):
        # compute average of exp(beta*V) on this simualtions
        coeff = 1 / torch.mean(weights[torch.nonzero(new_labels == i, as_tuple=True)])
        
        # update the weights
        weights[torch.nonzero(new_labels == i, as_tuple=True)] = coeff * weights[torch.nonzero(new_labels == i, as_tuple=True)]
    
    # update dataset
    dataset['weights'] = weights
    dataset['labels'] = new_labels

    return dataset

# %% [markdown]
# # Generate Descriptors
# 
# The following functions are for calculating the descriptors, You can customize this part according to the system, just have the function that returns descriptors d and the coresponding derivative to original Cartesian coordinates $\frac {\partial d}{\partial x}$

# %% [markdown]
# ## Get Derivatives

# %%

def getderiv(derivename,load_args,num_d,atomnum,back):
    fullderiv = torch.ones((1,atomnum*3,num_d))
    itter=0
    for dir in derivename:
        print(f'collecting {itter}/{len(derivename)}')
        start,stop,stride=load_args[itter]['start'],load_args[itter]['stop'],load_args[itter]['stride']
        itter+=1

        frameuse = np.arange(start,stop,stride)
        # print(frameuse)
        if back != 0:
            derivtmplarge = np.loadtxt(dir,skiprows=1,max_rows=stop*(atomnum*3+12))[:,-num_d-back:-back]
        else:
            derivtmplarge = np.loadtxt(dir,skiprows=1,max_rows=stop*(atomnum*3+12))[:,-num_d:]
        
        for i in frameuse:
            derivtmp = derivtmplarge[int(i*(atomnum*3+12)):int(i*(atomnum*3+12)+atomnum*3),:]
            # print(derivtmp.shape)
            tempderiv = torch.ones((1,atomnum*3,num_d))
            for j in range(num_d):
                derivdescrip_j  = derivtmp[:,j]

                tempderiv[0,:,j]=torch.from_numpy(derivdescrip_j)
            # print(tempderiv)
            fullderiv=torch.cat((fullderiv,tempderiv),0)
            # print(fullderiv.shape)
    fullderiv=fullderiv[1:,]
    print(fullderiv.shape)
    # print(fullderiv)
    return fullderiv


def getderiv_force(derivename,load_args,num_d,atomnum,back):
    fullderiv = torch.ones((1,atomnum*3,num_d))
    itter=0
    for dir in derivename:
        print(f'collecting {itter}/{len(derivename)}')
        start,stop,stride=load_args[itter]['start'],load_args[itter]['stop'],load_args[itter]['stride']
        itter+=1

        frameuse = np.arange(start,stop,stride)
        # print(frameuse)
        derivtmplarge = -np.loadtxt(dir)

        for i in frameuse:
            ##########  dU/dx=-F
            derivtmp = derivtmplarge[int(i*(atomnum)):int((i+1)*(atomnum))]
            derivtmp=derivtmp.reshape(-1)
            derivtmp=derivtmp.reshape(-1,1)
            # print(derivtmp.shape)
            tempderiv = torch.ones((1,atomnum*3,num_d))
            # print(tempderiv.shape)
            for j in range(num_d):
                derivdescrip_j  = derivtmp[:,j]

                tempderiv[0,:,j]=torch.from_numpy(derivdescrip_j)
            # print(tempderiv)
            fullderiv=torch.cat((fullderiv,tempderiv),0)
            # print(fullderiv.shape)
    fullderiv=fullderiv[1:,]
    print(fullderiv.shape)
    # print(fullderiv)
    return fullderiv

# %% [markdown]
# ## Define the cv to grid function for 2D plot
# it should be mentioned that the range should be set acoordingly based on the system you simulate

# %%
def compute_cv_on_grid(model):
    limits=((0,200),(20,60))
    num_points=(100,100)

    xx = np.linspace(limits[0][0],limits[0][1],num_points[0])
    yy = np.linspace(limits[1][0],limits[1][1],num_points[1])
    xv, yv = np.meshgrid(xx, yy)

    z = torch.zeros(num_points,device=device)
    for i in range(num_points[0]):
        for j in range(num_points[1]):
            xy = torch.Tensor([xv[i,j], yv[i,j]]).unsqueeze(0)
            xy = xy.to(device)
            z[i,j] = model(xy)
    return xv,yv,z

# %% [markdown]
# ## Define training process

# %%
def training(iter_id,cvbase,derivebase,grouplist,massdir,load_args,Loadnew):
    dnum=17

    ## Load data
    filenames =cvbase
    derivenames =derivebase
    # derivenamesf = [i+'forces.dump2' for i in dirbase]
    mixing = False
    mixing_csi = 0.5
    #######################################################################################

    dataset, dataframe = create_dataset_from_files(file_names = filenames,
                                                        folder = folder,
                                                create_labels = True,
                                                filter_args = {'regex': r'^laq6_ins.mean$|^laq6_ins.morethan$|^q6_ins.between-1$|^q6_ins.between-2$|^q6_ins.between-3$|^q6_ins.between-4$|^q6_ins.between-5$|^s2_ins.between-1$|^s2_ins.between-2$|^s2_ins.between-3$|^s2_ins.between-4$|^s2_ins.between-5$|^laq6_ins.between-1$|^laq6_ins.between-2$|^laq6_ins.between-3$|^laq6_ins.between-4$|^laq6_ins.between-5$'},
                                                return_dataframe = True,
                                                load_args=load_args,
                                                verbose = True)

    # compute the weight for the reweighing depending on the simualtions
    if 'bias' in dataframe.columns:
        dataframe = dataframe.fillna({'bias': 0})
    else:
        dataframe['bias'] = 0
    bias_tensor = torch.tensor(dataframe['bias'].values)
    # compute the weight for the reweighing depending on the simualtions
    # dataframe, dataset = compute_weights_mod(dataframe=dataframe, dataset=dataset, 
    #                                          beta=beta, mixing=mixing, mixing_csi=mixing_csi ,factor=1)
    dataset = compute_committor_weights(dataset, 
                            bias=bias_tensor, 
                            data_groups=grouplist, 
                            beta=beta)
    
    plt.plot(np.log10(dataset['weights'][:].cpu().detach().numpy()),label='newdat')
    plt.legend()
    plt.savefig(f'weight{iter_id}.png')
    plt.close()

    derivefull= getderiv(derivenames,load_args,num_d=dnum,atomnum=2916,back=0) 
    
    print (dataset['weights'])
    dataset['derivefull']=derivefull

    # print(dataset['derivefull'])
    print(dataset)
    torch.save(dataset,'datasetsmall.pt')    
    


# %% [markdown]
# ## Get descriptors based on their normalization parameters

# %%
def getparas():
    paras=np.loadtxt('normalize.para')
    for i in range(paras.shape[0]):
        print(f'd{i+1}: CUSTOM ARG=sq{i+1}.mean var=z PERIODIC=NO FUNC=(z-{paras[i,0]})/{paras[i,2]}')

# getparas()

# %% [markdown]
# # Workflow Iterations

# %% [markdown]
# ## Set T and $k_B$

# %% [markdown]
# It should be noticed that the program here is totally correct. However, for the **Plumed.dat** in bias simulations, the $\lambda$ should be set to $\lambda/\beta$ which is the true value of bias.

# %%
# temperature in Kelvin
T = 67 
# Boltzmann factor in the RIGHT ENRGY UNITS!
kb = 0.0083144621
beta = 1/(kb*T)
print(f'Beta: {beta} \n1/beta: {1/beta}')

# %% [markdown]
# It should be note that for the **initial iteration** the samples are unbiased and may cause difficult in realizing the phase transition with PLUMED. Therefore we can try using the **jitfile before-ahead** which are not converged which will show larger gradient to drive the phase transition simulations.
# However, from the **first iteration** we should still use the **final pt** as bias.

# %% [markdown]
# ### Do sampling with the bias 
# $$V(x) = - \lambda \log[|\nabla q(x) |^2 + \epsilon] - \log[\epsilon] \qquad \text{with} \quad \epsilon=1e-6$$
# 
# The second term is just to shift the bias to zero in the metastable basins

# %% [markdown]
# ## FROM Iter 0 or any
# ## Initial settings

# %%

## Load data
################################### SET THINGS HERE ###################################
folder = None
dirbaseinit =  [
    '/work/ydeng/lj/opes+committor/dataset/initial/liquid3/',
    '/work/ydeng/lj/opes+committor/dataset/initial/crystal3/',
             ]

cvbase=         [i+'COLVAR' for i in dirbaseinit]
derivbase=         [i+'deriv' for i in dirbaseinit]


load_args = [
    {'start' : 200, 'stop': 1000, 'stride': 5},
    {'start' : 200, 'stop': 1000, 'stride': 5},

            ]             


# %% [markdown]
# ## Start iterations

# %%
massdir='/work/ydeng/lj/opes+committor/dataset/initial/'
newcvbase =  [
    # these samples are reweighted based on the opes.bias+committor.bias+uppwall.bias
'/work/ydeng/lj/opes+committor/cm_run/iter5/m3rs/m3s3p1/COLVAR',
'/work/ydeng/lj/opes+committor/cm_run/iter5/m3rs/m3s3p2/COLVAR',
'/work/ydeng/lj/opes+committor/cm_run/iter5/m3rs/m3s3p3/COLVAR',
'/work/ydeng/lj/opes+committor/cm_run/iter5/m3rs/m3s3p4/COLVAR',
'/work/ydeng/lj/opes+committor/cm_run/iter5/m3rs/m3s4p1/COLVAR',
'/work/ydeng/lj/opes+committor/cm_run/iter5/m3rs/m3s4p2/COLVAR',
'/work/ydeng/lj/opes+committor/cm_run/iter5/m3rs/m3s4p3/COLVAR',
'/work/ydeng/lj/opes+committor/cm_run/iter5/m3rs/m3s4p4/COLVAR',
             ]

newderivebase =  [
'/work/ydeng/lj/opes+committor/cm_run/iter5/m3rs/m3s3p1/deriv',
'/work/ydeng/lj/opes+committor/cm_run/iter5/m3rs/m3s3p2/deriv',
'/work/ydeng/lj/opes+committor/cm_run/iter5/m3rs/m3s3p3/deriv',
'/work/ydeng/lj/opes+committor/cm_run/iter5/m3rs/m3s3p4/deriv',
'/work/ydeng/lj/opes+committor/cm_run/iter5/m3rs/m3s4p1/deriv',
'/work/ydeng/lj/opes+committor/cm_run/iter5/m3rs/m3s4p2/deriv',
'/work/ydeng/lj/opes+committor/cm_run/iter5/m3rs/m3s4p3/deriv',
'/work/ydeng/lj/opes+committor/cm_run/iter5/m3rs/m3s4p4/deriv',

             ]

load_argsnew = [
{'start' : 0, 'stop': 1800, 'stride': 10}, 
{'start' : 0, 'stop': 1800, 'stride': 10}, 
{'start' : 0, 'stop': 1800, 'stride': 10}, 
{'start' : 0, 'stop': 1800, 'stride': 10}, 
{'start' : 0, 'stop': 1800, 'stride': 10}, 
{'start' : 0, 'stop': 1800, 'stride': 10}, 
{'start' : 0, 'stop': 1800, 'stride': 10}, 
{'start' : 0, 'stop': 1800, 'stride': 10}, 

            ]  

load_args.extend(load_argsnew)
cvbase.extend(newcvbase)
derivbase.extend(newderivebase)
grouplist=[0,1,2,2,2,2,3,3,3,3]



iterstart=_iterid_

for i in range (1):

    print(cvbase)
    print(derivbase)

    print(len(cvbase))
    print(len(load_args))

    training(iter_id=iterstart,cvbase=cvbase,derivebase=derivbase,grouplist=grouplist,massdir=massdir,load_args=load_args,Loadnew=Loading_bool)   # True  False




