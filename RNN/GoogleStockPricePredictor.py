"""OSS: In questo codice non stiamo facendo train e validation, stiamo solo ponderando il train"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

LSTM_UNITS = 50
EPOCHS = 100

def plotLossCurves(history):
    plt.plot(history.history['loss'], label='Loss')
    plt.title('Curva di Errore durante l\'allenamento')
    plt.xlabel('Epoche')
    plt.ylabel('Errore (Loss)')
    plt.legend()
    plt.grid()
    plt.show()

train_dir = r"C:\Users\marcd\Desktop\projects\Corso_IA\Deep Learning A-Z\Part 3 - Recurrent Neural Networks (RNN)\Google_Stock_Price_Train.csv"


dataset_train = pd.read_csv(train_dir)
training_set = dataset_train.iloc[:,1:2].values

#iloc mi permete di selezionare porzioni di dati da un dataframe 
#.values mi permette di convertire i valori selezionati in un array numpy

sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

#Effettuo una trasformazione mim-max per normalizzare i valori
"""
Tale pratica è comune per permettere una convergenza migliore all'algorito di gradient
descend, in particolare quando si normalizza si porta anche l'insieme di valori a media
nulla e varianza 1.
"""

X_train = []
y_train = []

for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

"""
Le RNN si basano su serie temporali, in questo caso stiamo costruendo una struttura
che ogni 60 campioni precedenti, deve effettuare una predizione.
In tale caso ho un serie di serie temporali e alla prima serie ad esempio, associo la prima predizione di y_train
"""

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
"""
In questo caso necessitiamo un reshaping poichè il modello richiede questo tipo di struttura 
In particolare stiamo costruendo un array che ha come prima dimensione a lunghezza dell'array in righe e la seconda 
la lunghezza dell'array in colonne 
"""

regressor = Sequential()
"""
Difatti stiamo facendo una regressione lineare, su sequenze temporali
"""
regressor.add(LSTM(units = LSTM_UNITS, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

"""
Primo layer della rete, deve tornare una sequenza anche lui da passare al layer successivo, solo l'ultimo
layer non deve tornare una sequenza
Aggiungiamo il livello di dropout per evitare l'overfitting della rete
Durante l'allenamento, i neuroni di un layer vengono "disattivati" casualmente
(ossia, il loro output è impostato a zero) con una certa probabilità p, chiamata rate di dropout.
"""
regressor.add(LSTM(units = LSTM_UNITS, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = LSTM_UNITS, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = LSTM_UNITS))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

history = regressor.fit(X_train, y_train, epochs = EPOCHS, batch_size = 32)

plotLossCurves(history)

dataset_test = pd.read_csv(r'C:\Users\marcd\Desktop\projects\Corso_IA\Deep Learning A-Z\Part 3 - Recurrent Neural Networks (RNN)\Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

"""
Dobbiamo fare una serie di predizioni, però il formato di ingresso deve essere lo stesso
In questo caso dunque gli devo passare delle serie temporali.
Per farlo tenendo conto anche del periodo precedente (presente nel trainset) li devo concatenare.
almeno per gli ultimi campioni.
"""
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
print(predicted_stock_price)
