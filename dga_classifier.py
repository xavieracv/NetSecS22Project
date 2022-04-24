# -*- coding: utf-8 -*-
"""dga_classify_matt.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pOvnvwd54CsWGKxDzIVVfAp9UVIGFDpA
"""

#import glob
#from   google.colab import drive
import tensorflow as tf
import numpy as np
import pandas as pd
from   tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.layers import StringLookup
from keras.preprocessing import text
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

#drive.mount("/content/gdrive")

dataframe = pd.read_csv("dga_domains_full.csv",header=None, usecols=[2,0])

#print(dataframe.shape[0])

dataframe[0].replace(to_replace=['dga', 'legit'], value=[1,0], inplace=True)

print(dataframe.head())
print("\n")

X = dataframe[2] # domains
Y = dataframe[0] # label

#print(X.head())
#print(Y.head())

# domains + labels
# X_train + Y_train, X_test  + Y_test

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=23)

#for i in range(0,10):
#  print(str(X_train.values[i]) + " " + str(Y_train.values[i]))
#  print(str(X_test.values[i]) + " " + str(Y_test.values[i]))

# X_train = domains
# Y_train = labeles (1 or 0)

encoder = text.Tokenizer(num_words=100, char_level=True, oov_token='UNK') # char_level true is important
encoder.fit_on_texts(X_train)
X_train_sequences = encoder.texts_to_sequences(X_train) # convert all text to series of integers; now we have a list of lists
input_dim = len(encoder.word_index)+1
## For Verifying the vocabulary that was generated...
#print(input_dim)
#for k,v in sorted(encoder.word_index.items()):
#  print(str(k) + ":" +str(v))

X_train_sequences = pad_sequences(X_train_sequences, maxlen=255, padding='post')
X_train_sequences = np.array(X_train_sequences)
#print(min(map(len, X_train_sequences)))
#print(X_train_sequences[0:10]) # encoded version of drqnotyomfhgeso.net

# Build the model

model=Sequential()
model.add(Embedding(input_dim, 128, input_length=255))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop')
print("\nMODEL SUMMARY:")
print(model.summary())

batch_size = 128                                         

model.fit(X_train_sequences, Y_train, batch_size=batch_size, epochs=3)

