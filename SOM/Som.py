import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from minisom.minisom import MiniSom
from pylab import bone,pcolor,colorbar,plot,show

dataset = pd.read_csv(r"C:\Users\marcd\Desktop\projects\Corso_IA\Deep Learning A-Z\Part 4 - Self Organizing Maps (SOM)\Credit_Card_Applications.csv")

X = dataset.iloc[:,:-1]
Y = dataset.iloc[:,-1]

#Devo eseguire più o meno sempre lo scaling per permettere al modello di effettuare calcoli più agilmente 

sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)

#La input_len indica le features del dataset in ingresso 
som = MiniSom(x= 10, y = 10, input_len=15,sigma=1.0,learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)

"""
Allenando la SOM sono stati trovati i winning node ai quali vengono associati 
poi gli altri record della rete, ricordiamo che un winning node e` un record che piu` si avvicina
ad un neurone della rete e al quale verranno associati i record vicini 

Ad ogni winning node dunque viene associata una distanza, piu` questa e` altra piu`
Quest`ultima rappresenta una condizione di outlier
"""
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[Y[i]],
         markeredgecolor = colors[Y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

"""
Per intercettare quelli che sono i record che sono potenzialmente faudolenti devo 
prelevare tutti i record associati a winning node con distanza molto alta (nel nostro caso 1)
Una volta fatto questo dobbiamo effettuare una inverse transform poiche` abbiamo scalato all`inizio
"""
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8,1)], mappings[(6,9)]), axis = 0)
frauds = sc.inverse_transform(frauds)
print(frauds)


"""
ATT: ad ogni iterazione la SOM potrebbe essere riorganizzata dunque il metodo di sopra non e` del tutto generale, ma potrebbe essere automatizzato considerando la distanza
"""