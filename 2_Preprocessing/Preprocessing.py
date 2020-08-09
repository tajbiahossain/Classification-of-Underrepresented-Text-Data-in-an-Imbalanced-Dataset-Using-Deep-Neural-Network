# re - regular expression/ regex : used for text cleaning in python

import re
import nltk
import pandas as pd
from textblob import Word
import numpy as np
import csv
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

#function for text cleaning
def clean_str(string):
    string = re.sub(r"\'s", "", string)
    string = re.sub(r"\'ve", "", string)
    string = re.sub(r"n\'t", "", string)
    string = re.sub(r"\'re", "", string)
    string = re.sub(r"\'d", "", string)
    string = re.sub(r"\'ll", "", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"'", "", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"[0-9]\w+|[0-9]","", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()
    
data = pd.read_csv('bbc_raw.csv')


x = data['text'].tolist()
y = data['category'].tolist()
z = data['text'].tolist()
for index,value in enumerate(x):
    print ("preprocessing article number:",index)
    # text cleaning, lemmatization, and stopword removal
    x[index] = ' '.join([Word(word).lemmatize() for word in clean_str(value).split() if word not in set(stopwords.words('english'))])
    
data = {'category': y, 'text': x}       
df = pd.DataFrame(data)
df.to_csv('F:/AUST/Thesis/Predefence Codes (all)/Predefence Codes (all)/
          2_Preprocessing/bbc_preprocessed_lemm_stopwordremoved.csv', index=False)


