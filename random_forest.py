# # from tensorflow.keras.preprocessing.text import Tokenizer
# from keras import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
import pandas as pd

import pickle

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
  data_unpadded["tokenized"] = [ flatten(tokenizer.texts_to_sequences(domain)) for domain in data["domain"]]
  
  #find longest sequence created
  #maxLen = 0
  for sequ in data_unpadded["tokenized"]:
    l = len(sequ)
    if l > maxLen:
      maxLen = l
  
  data_padded = preprocessing.sequence.pad_sequences(data_unpadded["tokenized"], maxlen= maxLen)

  return np.array(data_padded), maxLen

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

def main():

  print("Loading CSV Data...")
  train = pd.read_csv("datasets/dns_train.csv")
  test = pd.read_csv("datasets/dns_test.csv")
  val = pd.read_csv("datasets/dns_val.csv")
  print("CSV Data Loaded.")

  # Merge train and test into large training dataset
  train = pd.concat([train, test], axis=0)

  #Change labels to binary 0 (NO TUNNELING) 1(TUNNELING)
  train_labels = np.asarray(train.pop("label"))
  train_labels = np.where(train_labels>1,1,train_labels)

  val_labels  =  np.asarray(val.pop("label"))
  val_labels = np.where(val_labels>1,1,val_labels)

  # Separate data from labels, drop domain
  domain_train = train.copy()
  domain_train = domain_train.drop("qname", axis=1)

  print(domain_train.info())

  domain_val = val.copy()
  domain_val = domain_val.drop("qname", axis=1)

  """ Pre-processing """
  pipeline = Pipeline([
      ('imputer', SimpleImputer(strategy='median')),
      ('std_scaler', StandardScaler(),)
  ])

  train_data = pipeline.fit_transform(domain_train)
  val_data = pipeline.transform(domain_val)    # which is actually the test data

  #Define Model
  model = RandomForestClassifier(random_state=36)
  
  # Train model and tune model
  # Potential scoring metrics: 
  #   https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
  print("Fitting model...")
  param_distribs = {
      'n_estimators': randint(low=1, high=10),
      'max_features': randint(low=1,  high=train_data.shape[1]),
      'max_depth':    randint(low=1,  high=2)
    }
  search = RandomizedSearchCV(
      model, 
      param_distributions=param_distribs,
      n_iter=100,
      cv=10,
      scoring='f1',
      random_state=36
  )
  search.fit(train_data, train_labels)

  best_model = search.best_estimator_

  print("Training score of model with params:", search.best_params_, "is", search.best_score_)

  print("Full model parameters:", best_model)

  print("Feature importances:")
  print(sorted(zip(best_model.feature_importances_, domain_train.columns.values.tolist()), reverse=True))

  print("Model performance metrics: ")

  y_pred = best_model.predict(val_data) 
  print("Precision: ", precision_score(val_labels, y_pred , average="macro"))
  print("Recall:    ", recall_score(val_labels, y_pred , average="macro"))
  print("F1 Score:  ", f1_score(val_labels, y_pred , average="macro"))
  print("Confusion Matrix:")
  print(confusion_matrix(val_labels, y_pred))
  
  return best_model


if __name__ == "__main__":
  dns_model = main()

  # Save model to current directory using pickle
  pkl_filename = "RANDOM_FOREST_DNS_PACKETS.pkl"
  print("Saving DNS Tunneling Model as", pkl_filename)
  with open(pkl_filename, 'wb') as outfile:
      pickle.dump(dns_model, outfile)
  
#   # Loading the model from the pickle file
#   with open(pkl_filename, 'rb') as infile:
#       pickle_model = pickle.load(infile)
