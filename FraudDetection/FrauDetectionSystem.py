"""# -*- coding: utf-8 -*-
Fraud detection system
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from minisom.minisom import MiniSom
from pylab import bone,pcolor,colorbar,plot,show

from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score

def printSom(som: MiniSom,X,Y):
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
    
def trainSom(som:MiniSom,X):
    som.random_weights_init(X)
    som.train_random(data=X, num_iteration=100)
    return som

def detectFrauds(som:MiniSom,sc,fraudDistanceDetect=0.95):
    mappings = som.win_map(X)
    fraud_arr = []
    matrix = som.distance_map()
    frauds = []
    for x,r in enumerate (matrix):
        for y,j in enumerate(r):
            if(j > fraudDistanceDetect):
               frauds.append((x,y)) 
    for element in frauds: 
        fraud_arr.append(mappings[element])
    frauds = np.concatenate(fraud_arr, axis = 0)
    frauds = sc.inverse_transform(frauds)
    return frauds
               
def printAccuracyPlot(history):
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(1, len(train_accuracy) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracy, label='Training Accuracy')
    plt.plot(epochs, val_accuracy, label='Validation Accuracy')
    plt.xlabel('Epoche')
    plt.ylabel('Accuracy')
    plt.title('Accuracy in funzione delle epoche')
    plt.legend()
    plt.grid(True)
    plt.show()

dataset = pd.read_csv(r"C:\Users\marcd\Desktop\projects\Corso_IA\Deep Learning A-Z\Part 4 - Self Organizing Maps (SOM)\Credit_Card_Applications.csv")

X = dataset.iloc[:,:-1]
Y = dataset.iloc[:,-1]

sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)
som = MiniSom(x= 10, y = 10, input_len=15,sigma=1.0,learning_rate=0.5)                   
som = trainSom(som,X)
printSom(som, X, Y)
frauds = detectFrauds(som,sc,0.95)
print(frauds)

#SUPERVISED SECTION
fraudolentCustomers = frauds[:,0]
dataset["fraudolent"] = dataset["CustomerID"].isin(fraudolentCustomers).astype(int)

X_supervised = dataset.iloc[:,:-2]
Y_supervised = dataset.iloc[:,-1]
X_supervised = sc.fit_transform(X_supervised)
X_train, X_test, y_train, y_test = train_test_split(X_supervised, Y_supervised, test_size = 0.2, random_state = 0)

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
history = ann.fit(X_train, y_train, batch_size = 32, epochs = 100,validation_data=(X_test, y_test))

printAccuracyPlot(history)


y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(y_pred)

cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
print(cm)