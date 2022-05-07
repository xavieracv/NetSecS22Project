from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, layers, models
from keras            import preprocessing
from keras.datasets   import reuters
from keras.models     import Sequential 
from keras.layers     import Dense, Dropout, Activation
from sklearn.metrics  import f1_score, precision_score, recall_score, confusion_matrix

import tensorflow as tf
import numpy      as np
import pandas     as pd

def preprocess(data, maxLen = 0):
  #print(data.head(5))
  tokenizer = Tokenizer(num_words=None, char_level=True)
  tokenizer.fit_on_texts(data)
  data_unpadded = pd.DataFrame()
  data_unpadded["tokenized"] = tokenizer.texts_to_sequences(data)
  
  #find longest sequence created
  #maxLen = 0
  for sequ in data_unpadded["tokenized"]:
    l = len(sequ)
    if l > maxLen:
      maxLen = l
  
  data_padded = preprocessing.sequence.pad_sequences(data_unpadded["tokenized"], maxlen=maxLen)

  return np.array(data_padded), maxLen

# Converts dataset into train and test split (modify this if the dataset changes)
def parse_csv():
    print("Loading CSV Data...")
    dns_train = pd.read_csv("datasets/dns_train.csv", usecols = [0,1])
    dns_test  = pd.read_csv("datasets/dns_test.csv",  usecols = [0,1])
    #dns_val   = pd.read_csv("datasets/dns_val.csv",  header = 1, usecols = [0,1])
    print("CSV Data Loaded.")

    # print(dns_train.shape)
    # print(dns_train.head())
    # print()
    # print(dns_test.shape)
    # print(dns_test.head())
    # print()

    dns_full = pd.concat([dns_train, dns_test], ignore_index=True)

    # print(dns_full.shape)
    # print(dns_full.head())
    # print()

    X = dns_full['qname']
    Y = dns_full['label'].map(lambda x: 1 if (x > 0) else 0)

    # for i in range(0, X.size):
    #     print(Y[i], end=', ')
    #     print(X[i])

    return train_test_split(X, Y, test_size=0.2,random_state=23)


# Build and Save LSTM model
def main():
    x_train, x_test, y_train, y_test = parse_csv()

    MAX_LENGTH = 255 # DOMAINS ARE CAPPED AT 255 (including '.')

    print(x_train[0])

    # process domains into sequences that our model can understand
    x_train,  _ = preprocess(x_train, MAX_LENGTH)
    #y_train, _ = preprocess(y_train, MAX_LENGTH)


    print(x_train[0])

    

if __name__=="__main__":
    main()