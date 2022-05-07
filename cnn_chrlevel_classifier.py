

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from keras import preprocessing
from keras.datasets import reuters
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Activation
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
import pandas as pd

"""
Preprocess Data: Tokenize and Pad
Functions to Preprocess Data
"""
def flatten(t):
    return [item for sublist in t for item in sublist]

def preprocess(data, maxLen = 0):
  #print(data.head(5))
  tokenizer = Tokenizer(num_words=None, char_level=True)
  tokenizer.fit_on_texts(data["domain"])
  data_unpadded = pd.DataFrame()
  data_unpadded["tokenized"] = [ tokenizer.texts_to_sequences(domain) for domain in data["domain"]]
  
  #find longest sequence created
  #maxLen = 0
  for sequ in data_unpadded["tokenized"]:
    l = len(sequ)
    if l > maxLen:
      maxLen = l
  
  data_padded = preprocessing.sequence.pad_sequences(data_unpadded["tokenized"], maxlen= maxLen)

  return np.array(data_padded), maxLen




def main():

  print("Loading CSV Data...")
  train = pd.read_csv("datasets/dns_train.csv")
  test = pd.read_csv("datasets/dns_test.csv")
  val = pd.read_csv("datasets/dns_val.csv")
  print("CSV Data Loaded.")

  #Separate domain data from label;s
  domain_train = pd.DataFrame()
  domain_train["domain"] = train.pop("qname")

  domain_test = pd.DataFrame()
  domain_test["domain"]= test.pop("qname")

  domain_val = pd.DataFrame()
  domain_val["domain"] = val.pop("qname")

  #Change labels to binary 0 (NO TUNNELING) 1(TUNNELING)
  train_labels = np.asarray(train.pop("label"))
  train_labels = np.where(train_labels>1,1,train_labels)

  test_labels  =  np.asarray(test.pop("label"))
  test_labels = np.where(test_labels>1,1,test_labels)

  val_labels  =  np.asarray(val.pop("label"))
  val_labels = np.where(val_labels>1,1,val_labels)


  """Preprocess training and test data"""

  maxLen_of_both = 253  # of both train and test

  print("Preprocessing Data Sets...")
  train_data, _ = preprocess(domain_train, maxLen_of_both)
  val_data,_ = preprocess(domain_val,maxLen_of_both)
  test_data, maxLen_of_test = preprocess(domain_test, maxLen_of_both)
  print("Preprocessing done.")  

  print(train_data[0])


  """Reshape Data for Model Input"""
  row,col = train_data.shape
  train_data = tf.reshape(train_data,(row,col,1))

  r, c = test_data.shape
  test_data = tf.reshape(test_data,(r,c,1))

  r, c = val_data.shape
  val_data= tf.reshape(val_data,(r,c,1))


  #Define Model
  model = Sequential()
  model.add(layers.Input(shape=(col,1)))
  model.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu'))
  model.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu'))
  model.add(layers.MaxPooling1D(pool_size=2))
  model.add(layers.Flatten())
  model.add(layers.Dense(4000))
  model.add(layers.Dropout(0.3))
  model.add(layers.Dense(1, activation='sigmoid'))
  print(model.summary())
  model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])

  
  print("Fitting model...")
  model.fit(train_data,train_labels,validation_data=(test_data,test_labels),batch_size=128,epochs=3)


  print("Model performance metrics: ")

  y_pred = model.predict(val_data,verbose=1)
  Convert to binary classification (0/1)
  y_pred = (y_pred > 0.5)
  Replace bool value with 0/1
  y_pred = np.where(y_pred==True,1,y_pred)
  print("Precision: ",precision_score(val_labels, y_pred , average="macro"))
  print("Recall:    ", recall_score(val_labels, y_pred , average="macro"))
  print("F1 Score:  ",f1_score(val_labels, y_pred , average="macro"))
  
  return model



if __name__ == "__main__":
  dns_model = main()
  print("Saving DNS Tunneling Model as DNS_MODEL")
  dns_model.save("DNS_MODEL")
