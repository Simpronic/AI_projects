# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 14:27:54 2025

@author: marcd
"""

import torch 
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

class SAE(nn.Module):
    def __init__(self,number_of_features,hn_neurons, ):
        super(SAE,self).__init__()
        self.fc1 = nn.Linear(number_of_features, hn_neurons)
        self.fc2 = nn.Linear(hn_neurons, int(hn_neurons/2))
        self.fc3 = nn.Linear(int(hn_neurons/2), hn_neurons)
        self.fc4 = nn.Linear(hn_neurons, number_of_features)
        self.activation = nn.Sigmoid()
    # x e` il vettore delle features (in questo caso i film recensiti dall`utente)
    def forward(self,x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        #Non applico la funzione di attivazione poiche` e` il livello di output
        x = self.fc4(x)
        return x
        