import signal, time, random, itertools, copy, math, sys
import pickle as pkl
# numpy imports
import numpy as np
# torch imports
import torch
import torch.autograd as ta
import torch.nn.functional as F
import torch.nn as nn
# code imports

####
# Basic NN classes
####

class Emb(nn.Module):
    
    def __init__(self, emb_ct, emb_dim, device=torch.device('cpu'), 
                 sparse_grads=False):
        super().__init__()
        self.embedding = nn.Embedding(emb_ct + 1, emb_dim, padding_idx=emb_ct,
                                      sparse=sparse_grads).to(device)
        #nn.init.kaiming_uniform_(self.embedding.weight)

    def forward(self, x):
        return self.embedding(x)

class MLP(nn.Module):

    def __init__(self, inp_dim, out_dim, hid_dim=None, device=torch.device('cpu'),
                 mlp_act='relu', inner_act='relu', mlp_layers=2, dropout=False,
                 norm_type='batch'):
        super().__init__()
        
        def make_act(act):
            if act == 'relu': return nn.ReLU()
            elif act == 'tanh': return nn.Tanh()
            elif act == 'sigmoid': return nn.Sigmoid()
            elif act == 'elu': return nn.ELU()

        def make_norm(norm, dim):
            if norm == 'batch': return nn.BatchNorm1d(dim)
            elif norm == 'layer': return nn.LayerNorm(dim)
            elif norm == None: return nn.Identity(dim)

        if hid_dim == None: hid_dim = out_dim

        if dropout: drpt = nn.Dropout(dropout)
        
        modules = []
        if mlp_layers == 1:
            modules.append(nn.Linear(inp_dim, out_dim))
        else:
            for l in range(mlp_layers - 1):
                i_dim = inp_dim if l == 0 else hid_dim
                modules.append(nn.Linear(i_dim, hid_dim))
                modules.append(make_norm(norm_type, hid_dim))
                modules.append(make_act(inner_act))
                if dropout: modules.append(drpt)
            modules.append(nn.Linear(hid_dim, out_dim))
        if mlp_act != 'sigmoid':
            modules.append(make_norm(norm_type, out_dim))
        modules.append(make_act(mlp_act))
        if mlp_act != 'sigmoid' and dropout: modules.append(drpt)
        self.ff = nn.Sequential(*modules).to(device)
        
    def forward(self, x):
        return self.ff(x)

class EmbProj(nn.Module):
    # simple layered embedding
    def __init__(self, concept_ct, concept_emb_dim, concept_state_dim,
                 layer=False, device=torch.device('cpu'), sparse_grads=False):
        super().__init__()
        self.layer = layer
        if layer:
            self.emb_layer = Emb(concept_ct, concept_emb_dim, device=device,
                                 sparse_grads=sparse_grads)
            self.bn = nn.BatchNorm1d(concept_emb_dim).to(device)
            self.proj_layer = MLP(concept_emb_dim, concept_state_dim,
                                  mlp_layers=1, device=device)
        else:
            self.emb_layer = Emb(concept_ct, concept_state_dim, device=device,
                                 sparse_grads=sparse_grads)
            self.bn = nn.BatchNorm1d(concept_state_dim).to(device)
        self.act = nn.ReLU()

    def forward(self, x):
        if self.layer:
            return self.proj_layer(self.act(self.bn(self.emb_layer(x))))
        return self.act(self.bn(self.emb_layer(x)))

class NormalizeIntoLinear(nn.Module):

    def __init__(self, inp_dim, out_dim, bias=True, device=torch.device('cpu')):
        super().__init__()
        self.layer = nn.Sequential(*[nn.LayerNorm(inp_dim), nn.Linear(inp_dim, out_dim, bias=bias)])
        self.layer = self.layer.to(device)

    def forward(self, x):
        return self.layer(x)
        
###
# Skip connection
###

class SkipConnection(nn.Module):
    
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)

