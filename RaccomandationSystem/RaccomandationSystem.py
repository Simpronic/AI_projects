# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 12:11:09 2025

@author: marcd
"""

import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from SAE import SAE


def convert(nb_movies,nb_users,dataset):
    array_to_return = []
    for user in range(1,nb_users+1):
        movies = dataset[:,1][dataset[:,0] == user]
        id_rating = dataset[:,2][dataset[:,0] == user]
        record = np.zeros(nb_movies)
        record[movies-1] = id_rating
        array_to_return.append(record)
    return array_to_return

"""def train_sae(epochs,sae,n_batch,training_set,criterion,optimizer):
    new_sae = sae
    for epoch in range(1,epochs+1):
        train_loss = 0
        s = 0.
        for batch in range(n_batch):
            input = Variable(training_set[batch]).unsqueeze(0)
            target = input.clone()
            if torch.sum(target.data > 0) > 0:
                output = sae(input)
                target.require_grad = False
                output[target == 0] = 0
                loss = criterion(output,target)
                mean_corrector = len(target.data)/float(torch.sum(target.data > 0) + 1e-10)
                loss.backward()
                train_loss += np.sqrt(loss.data[0]*mean_corrector)
                s += 1.
                optimizer.step()
        print(f"epoch: {epoch}, loss: {train_loss/s}")
    return new_sae"""


movies = pd.read_csv('ml-1m/movies.dat',sep = '::',header = None, engine = 'python',encoding='latin-1')
users = pd.read_csv('ml-1m/users.dat',sep = '::',header = None, engine = 'python',encoding='latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat',sep = '::',header = None, engine = 'python',encoding='latin-1')

#Sono tutti numeri e lidevo trattare come array numerici, altrimenti lli trattera come dataframe 
training_set = pd.read_csv('ml-100k/u1.base',delimiter='\t')
training_set = np.array(training_set,dtype='int')
test_set = pd.read_csv('ml-100k/u1.test',delimiter='\t')
test_set = np.array(test_set,dtype='int')


nb_users = int(max(max(training_set[:,0]),max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]),max(test_set[:,1])))

training_set = convert(nb_movies,nb_users,training_set)
test_set = convert(nb_movies,nb_users,test_set)

#Ho bisogno di tensori poiche' tensor flow allena le reti Deep tramite una rappresentazione tensoriale 
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

sae = SAE(nb_movies,20)
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(),lr = 0.01, weight_decay=0.5)

#trained_sae = train_sae(200, sae, nb_users, training_set, criterion, optimizer)

nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.data*mean_corrector)
            s += 1.
            optimizer.step()
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

prediction_input = Variable(training_set[1]).unsqueeze(0)
target_pred = Variable(test_set[1]).unsqueeze(0)

test_loss = 0
output = sae(prediction_input)
target_pred.require_grad = False
output[target_pred == 0] = 0
loss = criterion(output, target_pred)
mean_corrector = nb_movies/float(torch.sum(target_pred.data > 0) + 1e-10)
test_loss += np.sqrt(loss.data*mean_corrector)

print(f"Target is {target_pred.data}")
print(f"Prediction is: {output.data} with loss: {test_loss}")
