# -*- coding: utf-8 -*-
"""Copy of DGA_classifier1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TK4n-J3zKnsmbz3sVKpwL6yTti7ZgLtt

##Import Tensorflow
"""

print("Loading dependencies...")
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

"""#Upload and Prep Data Set"""

#Dataset is in NetSec Google Drive Directory
import numpy as np
import glob
import pandas as pd
#from google.colab import drive
#drive.mount("/content/gdrive")

#path = "/content/gdrive/MyDrive/NetSec_S22_Project/classifier_dataset/data"

print("Loading CSV Data...")
domain_train = pd.read_csv("train_combined_multiclass.csv",header=[0])
domain_test = pd.read_csv("test_combined_multiclass.csv",header=[0])
print("CSV Data Loaded.")
train_labels = pd.DataFrame(domain_train.pop("class"))
test_labels  =  np.asarray(domain_test.pop("class"))

domain_train.head()

train_labels

classes = np.unique(train_labels)
n_classes = len(classes)
#print(classes)

domain_test.head()

"""##Preprocess Data: Tokenize and Pad

###Functions to Preprocess Data
"""

from tensorflow.keras.preprocessing.text import Tokenizer
from keras import preprocessing

def flatten(t):
    return [item for sublist in t for item in sublist]

def preprocess(data, maxLen = 0):
  tokenizer = Tokenizer(num_words=None, char_level=True)
  tokenizer.fit_on_texts(data["domain"])
  data_unpadded = pd.DataFrame()
  data_unpadded["tokenized"] = [ flatten(tokenizer.texts_to_sequences(domain)) for domain in data["domain"]]
  
  #find longest sequence created
  #maxLen = 0
  for sequ in data_unpadded["tokenized"]:
    l = len(sequ)
    if l > maxLen:
      maxLen = l
  
  data_padded = preprocessing.sequence.pad_sequences(data_unpadded["tokenized"], maxlen= maxLen)

  return np.array(data_padded), maxLen

"""###Preprocess training and test data"""

#train_data = pd.DataFrame()
#train_data, maxLen = preprocess(domain_train)
#test_data = pd.DataFrame()
#test_data, _ = preprocess(domain_test, maxLen)

maxLen_of_both = 253  # of both train and test

print("Preprocessing Train and Test Sets...")
train_data, _ = preprocess(domain_train, maxLen_of_both)
test_data, maxLen_of_test = preprocess(domain_test, maxLen_of_both)
assert(maxLen_of_test == maxLen_of_both)
print("Preprocessing done.")
"""##Create Model/Train"""

from keras.datasets import reuters
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Activation

rows = len(train_data)
cols = len(train_data[0])

train_data = tf.reshape(train_data,(rows,cols,1))

print((cols,1))
print((rows,cols,1))

print(test_data.shape)

r, c = test_data.shape
test_data = tf.reshape(test_data,(r,c,1))

from tensorflow.python.eager.context import collect_graphs

model = Sequential()
model.add(layers.Input(shape=(cols,1)))
model.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(layers.MaxPooling1D(pool_size=2))

model.add(layers.Flatten())
model.add(layers.Dense(1920, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(n_classes, activation='softmax'))

print(model.summary())


model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

print("Fitting model...")
model.fit(train_data,train_labels,validation_data=(test_data,test_labels),batch_size=128,epochs=3)

"""Note: previous error was occurring because the shape of the train and test data is not the same."""
model.save("DGAModel")