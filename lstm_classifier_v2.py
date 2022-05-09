from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, layers, models

# from keras            import preprocessing
# from keras.datasets   import reuters
# from keras.models     import Sequential 
#from keras.layers     import Dense, Dropout, Activation, Embedding, CuDNNLSTM
from sklearn.metrics  import f1_score, precision_score, recall_score, confusion_matrix
# import tensorflow as tf

from tensorflow.keras import Sequential, preprocessing
from tensorflow.keras.layers import Dense, Dropout, Activation, Embedding, LSTM
import numpy      as np
import pandas     as pd


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

# Converts dataset into train and test split (modify this if the dataset changes)
def parse_csv():
    print("Loading CSV Data...")
    dns_train = pd.read_csv("datasets/dns_train.csv", usecols = [0,1])
    dns_test  = pd.read_csv("datasets/dns_test.csv",  usecols = [0,1])
    dns_val   = pd.read_csv("datasets/dns_val.csv",   usecols = [0,1])
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

    X = dns_full['qname'].map(lambda x: x[:-1]) # remove ending period from domains
    Y = dns_full['label'].map(lambda x: 1 if (x > 0) else 0) # convert label to 0 or 1
    V = dns_val['label'].map(lambda x: 1 if (x > 0) else 0)
    # for i in range(0, X.size):
    #     print(Y[i], end=', ')
    #     print(X[i])

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=23)

    return  x_train, x_test, y_train, y_test, dns_val['qname'], V


# Build and Save LSTM model
def main():
    x_train, x_test, y_train, y_test, val_data, val_labels = parse_csv()  # x_train, x_test -> data; y_train, y_test -> labels

    MAX_LENGTH = 255 # DOMAINS ARE CAPPED AT 255 (including '.')

    #print(x_train.iloc[0:2])
    
    # process domains into sequences that our model can understand
    x_train,  MAX_LENGTH = preprocess(x_train, MAX_LENGTH) # Training set
    x_test,   MAX_LENGTH = preprocess(x_test, MAX_LENGTH) # Validation set
    val_data, MAX_LENGTH = preprocess(val_data, MAX_LENGTH)


    #print(val_data[0])
    #print(x_train[0:2])

    # print(x_train.shape)
    # x_train = x_train.reshape(x_train.shape[0], 1,  x_train.shape[1])
    # x_test = x_test.reshape(x_test.shape[0], 1,  x_test.shape[1])
    # val_data = val_data.reshape(val_data.shape[0], 1,  val_data.shape[1])
    # print(x_train.shape)


    #print(x_train[0])
    '''At this point our dataset is in proper format'''

    input_dim  = 66 # size of vocab; i.e., len(vocab) 
    output_dim = 64 # size of dense embedding; i.e., length of output vectors

    #model.add(Embedding(input_dim, 2, input_length=MAX_LENGTH))
    # print(x_train.shape)
    # print(x_train[0].shape)

    # Define Model
    model = Sequential()
    model.add(Embedding(input_dim, 128, input_length=MAX_LENGTH))
    model.add(LSTM(128))
    model.add(Dropout(0.5))

    # model.add(LSTM(64))
    # model.add(Dropout(0.2))

    # model.add(Dense(16, activation='relu'))
    # model.add(Dropout(0.2))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    print(model.summary())
    model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc']) # switching the optimizer from adam to rmsprop made huge difference 

    # print(model.predict(x_train[0]).shape)

    print("Fitting model...")
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test,y_test))
  
    print("Model performance metrics: ")

    y_pred = model.predict(val_data,verbose=1)

    # Convert to binary classification (0/1)
    y_pred = (y_pred > 0.5)

    # Replace bool value with 0/1
    y_pred = np.where(y_pred==True,1,y_pred)
    print("Precision: ", precision_score(val_labels, y_pred , average="macro"))
    print("Recall:    ", recall_score(val_labels, y_pred , average="macro"))
    print("F1 Score:  ", f1_score(val_labels, y_pred , average="macro"))

    print("Saving Model to disk")
    #model.save("../CuDNNLSTM_V1")

    model.save("LSTM_V2")

if __name__=="__main__":
    main()