import re
import nltk
import pandas as pd
from textblob import Word
import numpy as np
import csv
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

#preprocessed dataset as input - text cleaned, stopwords removed, lemmatized
data = pd.read_csv('bbc_preprocessed.csv')

x = data['text'].tolist()
y = data['category'].tolist()

#feature extraction using count vectorizer and tf-idf
cv=CountVectorizer()
word_count_vector=cv.fit_transform(x)
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)
vect = TfidfVectorizer(stop_words='english',min_df=2)
X = vect.fit_transform(x)
Y = np.array(y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print ("\ntrain size:", X_train.shape)
print ("\ntest size:", X_test.shape)

results=[]
models=[]

#decision tree classifier

from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(X_train, Y_train)
model_predictions = model.predict(X_test)
print("\n---------Decision tree classifier--------- ")
print("\nConfusion matrix: \n", confusion_matrix(Y_test,model_predictions))
print("\nClassification Report: \n", classification_report(Y_test,model_predictions))
print("\nClassification accuracy: " ,accuracy_score(Y_test, model_predictions) * 100, "%.\n")

results.append(accuracy_score(Y_test, model_predictions))
models.append('DT')
#KNN classifier

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, Y_train) 
model_predictions = model.predict(X_test)
print("\n---------KNN classifier---------")
print("\nConfusion matrix: \n", confusion_matrix(Y_test,model_predictions))
print("\nClassification Report: \n", classification_report(Y_test,model_predictions))
print("\nClassification accuracy: " ,accuracy_score(Y_test, model_predictions) * 100, "%.\n")
results.append(accuracy_score(Y_test, model_predictions))
models.append('KNN')


#random forest classifier

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=1000, random_state=0)
model.fit(X_train, Y_train) 
model_predictions = model.predict(X_test)
print("\n---------Random Forest classifier---------")
print("\nConfusion matrix: \n", confusion_matrix(Y_test,model_predictions))
print("\nClassification Report: \n", classification_report(Y_test,model_predictions))
print("\nClassification accuracy: " ,accuracy_score(Y_test, model_predictions) * 100, "%.\n")
results.append(accuracy_score(Y_test, model_predictions))
models.append('RF')

#Naive Bayes classifier

from sklearn import naive_bayes
model = naive_bayes.MultinomialNB()
model.fit(X_train, Y_train) 
model_predictions = model.predict(X_test)
print("\n---------Naive Bayes classifier---------")
print("\nConfusion matrix: \n", confusion_matrix(Y_test,model_predictions))
print("\nClassification Report: \n", classification_report(Y_test,model_predictions))
print("\nClassification accuracy: " ,accuracy_score(Y_test, model_predictions) * 100, "%.\n")
results.append(accuracy_score(Y_test, model_predictions))
models.append('NB')

#SVM classifier

from sklearn import svm
model = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
model.fit(X_train, Y_train) 
model_predictions = model.predict(X_test)
print("\n---------Support Vector Machine classifier---------")
print("\nConfusion matrix: \n", confusion_matrix(Y_test,model_predictions))
print("\nClassification Report: \n", classification_report(Y_test,model_predictions))
print("\nClassification accuracy: " ,accuracy_score(Y_test, model_predictions) * 100, "%.\n")
results.append(accuracy_score(Y_test, model_predictions))
models.append('SVM')

#Logistic regression classifier

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(verbose=1, solver='liblinear',random_state=0, C=5, penalty='l2',max_iter=1000)
model.fit(X_train, Y_train) 
model_predictions = model.predict(X_test)
print("\n---------Logistic regression classifier---------")
print("\nConfusion matrix: \n", confusion_matrix(Y_test,model_predictions))
print("\nClassification Report: \n", classification_report(Y_test,model_predictions))
print("\nClassification accuracy: " ,accuracy_score(Y_test, model_predictions) * 100, "%.\n")
results.append(accuracy_score(Y_test, model_predictions))
models.append('LogReg')

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

objects = models
y_pos = np.arange(len(objects))
performance = results
plt.xlim(0.7, 1.0)
plt.barh(y_pos, performance, align='center', alpha=0.5)
plt.yticks(y_pos, objects)
plt.ylabel('Algrithms')
plt.xlabel('Accuracy')
plt.title('Classification Algorithm Comparison')

plt.show()

