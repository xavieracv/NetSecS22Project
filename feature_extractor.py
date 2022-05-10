# from tensorflow.keras.preprocessing.text import Tokenizer
# from keras import preprocessing

import regex as re

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

def shannon_entropy(domain):
    # Calculate character frequencies
    frequencies = {}
    for c in domain:
        frequencies[c] = domain.count(c)
    
    # Calculate character probabilities
    probabilities = []
    for _, v in frequencies.items():
        probabilities.append(v/len(domain))
    probabilities = np.array(probabilities)

    # Calculate entropy
    entropy = np.sum(probabilities*np.log2(probabilities))
    return -1*entropy

def vowel_consonant_ratio(domain):
    # Extract letters
    p = re.compile("[a-zA-Z]")
    letters = p.findall(domain)
    if len(letters) == 0:
        return 0

    # loop over and count ones in set of vowels
    letters = [l.lower() for l in letters]
    num_vowels = len([v for v in letters if v in ["a","e","i","o","u"]])

    # divide num_vowels by len(domain) - num_vowels
    ratio = num_vowels / (len(letters) - num_vowels)
    return ratio

def length(domain):
    return len(domain)

def num_digits(domain):
    p = re.compile("\d")
    digits = p.findall(domain)
    return len(digits)

def num_special(domain):
    p = re.compile("[^a-zA-Z0-9]")
    specials = p.findall(domain)
    return len(specials)

def num_dashes(domain):
    p = re.compile("\-")
    dashes = p.findall(domain)
    return len(dashes)

def num_colons(domain):
    p = re.compile(":")
    colons = p.findall(domain)
    return len(colons)

def extract_features(domain_data):
    processed_data = pd.DataFrame()
    processed_data['shannon_entropy'] = domain_data['domain'].transform(shannon_entropy)
    processed_data['vowel_consonant_ratio'] = domain_data['domain'].transform(vowel_consonant_ratio)
    processed_data['domain_length'] = domain_data['domain'].transform(length)
    processed_data['num_digits'] = domain_data['domain'].transform(num_digits)
    processed_data['num_special'] = domain_data['domain'].transform(num_special)
    processed_data['num_dashes'] = domain_data['domain'].transform(num_dashes)
    processed_data['num_colons'] = domain_data['domain'].transform(num_colons)
    return processed_data

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

    # Obtain features from domain
    domain_train = extract_features(pd.DataFrame({"domain": train.pop("qname")}))

    print(domain_train.info())

    domain_val = extract_features(pd.DataFrame({"domain": val.pop("qname")}))

    """ Pre-processing """
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('std_scaler', StandardScaler(),)
    ])

    train_data = pipeline.fit_transform(domain_train)
    val_data = pipeline.transform(domain_val)    # which is actually the test data

    #Define Model
    model = RandomForestClassifier(n_estimators=100, random_state=36)

    # Train model and tune model
    # Potential scoring metrics: 
    #   https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    print("Fitting model...")
    param_distribs = {
        'n_estimators': randint(low=10, high=150),
        'max_features': randint(low=1,  high=train_data.shape[1]),
        'max_depth':    randint(low=1,  high=10)
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
    pkl_filename = "RANDOM_FOREST.pkl"
    print("Saving DNS Tunneling Model as", pkl_filename)
    with open(pkl_filename, 'wb') as outfile:
        pickle.dump(dns_model, outfile)