#deep set model file
#the permutation equivariant layer are from deep set author's github, paper address: https://arxiv.org/abs/1703.06114
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd
import h5py
import pdb
from tqdm import tqdm, trange


class PermEqui_max(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(PermEqui_max, self).__init__()
    self.Gamma = nn.Linear(in_dim, out_dim)

  def forward(self, x):
    xm, _ = x.max(1, keepdim=True)
    x = self.Gamma(x-xm)
    return x

class PermEqui_mean(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(PermEqui_mean, self).__init__()
    self.Gamma = nn.Linear(in_dim, out_dim)
    self.Lambda = nn.Linear(in_dim, out_dim,bias = False)

  def forward(self, x):
    xm = x.mean(1, keepdim=True)

    xm = self.Gamma(xm)
    x = self.Lambda(x)
    x = x - xm
    return x

class Deepset(nn.Module):
  def __init__(self, perm_layer_type = 'mean', min = False, fg = False, d_dim = 256, x_dim=26):
    super(Deepset, self).__init__()
    self.d_dim = d_dim
    self.x_dim = x_dim
    self.perm_layer_type = perm_layer_type
    self.min = min
    self.fg = fg
    if self.perm_layer_type == 'max':
      self.perm = nn.Sequential(
        PermEqui_max(self.x_dim, self.d_dim),
        nn.Tanh(),
        PermEqui_max(self.d_dim, self.d_dim), 
        nn.Tanh(),    
        # PermEqui_max(self.d_dim, self.d_dim), 
        # nn.Tanh(),    
      )
      self.lastlayer = PermEqui_max(self.d_dim, self.x_dim)
    else:
      self.perm = nn.Sequential(
#===================================================permutation equivariant stack======================================
# comment:
# PermEqui_mean(self.x_dim, self.d_dim),
# nn.Tanh(),
# to change the number of layers in the permutation equivariant stack
        PermEqui_mean(self.x_dim, self.d_dim),
        nn.Tanh(),
        PermEqui_mean(self.d_dim, self.d_dim),  
        nn.Tanh(), 
        PermEqui_mean(self.d_dim, self.d_dim),  
        nn.Tanh(),   
#======================================================================================================================
      )
      self.lastlayer = PermEqui_mean(self.d_dim, self.x_dim) 

    self.down_samplingpool = nn.MaxPool1d(2,stride = 2)
    self.up_sampling = nn.Upsample(scale_factor=2, mode='linear')
    

  def forward(self, x):
    x = self.perm(x) # ab ==> abab

    if self.fg:  
      x = x.transpose(1,2)      #12 x 25 ---> 6 x 25 --->25 x 6 ---> 25 x 3 ---> 3 x 25
      x = self.down_samplingpool(x)
      x = x.transpose(1,2)

    if self.min:
      x = x.transpose(1,2) # [1,256,8]
      x = self.up_sampling(x)
      x = x.transpose(1,2)

    x = self.lastlayer(x)
    
    return x


