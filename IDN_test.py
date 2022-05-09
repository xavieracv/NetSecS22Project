from tensorflow import keras
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import Sequential, preprocessing
import tensorflow as tf


# model = keras.models.load_model('../LSTM_V2')

def flatten(t):
    return [item for sublist in t for item in sublist]

def preprocess2(data, maxLen = 0):
  #print(data.head(5))
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

def preprocess(data, maxLen = 0):
    '''Prebuilt Alphabet for Domains (Note: starts at 1, so 0 will only be used for padding'''
    chr_dict = { 
      'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10, 
      'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 
      't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26, 'A': 27, 'B': 28, 
      'C': 29, 'D': 30, 'E': 31, 'F': 32, 'G': 33, 'H': 34, 'I': 35, 'J': 36, 'K': 37, 
      'L': 38, 'M': 39, 'N': 40, 'O': 41, 'P': 42, 'Q': 43, 'R': 44, 'S': 45, 'T': 46, 
      'U': 47, 'V': 48, 'W': 49, 'X': 50, 'Y': 51, 'Z': 52, 
      '0': 53, '1': 54, '2': 55, '3': 56, '4': 57, '5': 58, '6': 59, '7': 60, '8': 61, '9': 62, 
      '-': 63, '_': 64, '.': 65 #, 'UNK':66
    }


    tokenizer = Tokenizer(num_words=None, char_level=True)
    tokenizer.word_index = chr_dict
    tokenizer.word_index[tokenizer.oov_token] = len(chr_dict) + 1
    #print(tokenizer.word_index)

    ''' expecting data to be pd series '''

    # find longest sequence created
    for sequ in data:
      l = len(sequ)
      if l > maxLen:
        maxLen = l

    print("MAXLENGTH = " + str(maxLen))

    data = tokenizer.texts_to_sequences(data)
    data = preprocessing.sequence.pad_sequences(data, maxlen=maxLen, padding='post') # make all domain encodings equal length

    return np.array(data), maxLen



IDN = pd.read_csv("datasets/IDN.csv")
domains, _ = preprocess(IDN['qname'], maxLen=255)
#domains = domains.reshape(domains.shape[0], 1,  domains.shape[1])
labels  = np.array(IDN['label'])

print(domains[:5])
print(labels[:5])

# load model
model = keras.models.load_model('../LSTM_V2')
model.summary()
score = model.evaluate(domains, labels, verbose = 0)


# Print Accuracy
print("LSTM CHRLEVEL")
print("test loss, test acc:", score)



domain_test = pd.DataFrame()
domain_test["domain"]= IDN.pop("qname")

test_data, maxLen_of_test = preprocess2(domain_test, 254)

r, c = test_data.shape
test_data = tf.reshape(test_data,(r,c,1))



# print(test_data.shape)
# print(test_data[0])

# print(labels.shape)
# print(labels[0])

model2 = keras.models.load_model('../CNN_CHRLVEL_DNS')
model2.summary()
score = model2.evaluate(test_data, labels, verbose = 0)

# Print Accuracy
print("CNN CHRLEVEL")
print("test loss, test acc:", score)