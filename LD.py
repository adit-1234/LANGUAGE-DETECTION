import numpy as numpy
import pandas as pd
import string
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import feature_extraction
from sklearn import pipeline
from sklearn import linear_model
from sklearn import metrics

dh=pd.read_csv('Language Detection.csv')
print(dh.head())


def remove_pun(text):
    for pun in string.punctuation:
        text.replace(pun,"")
    text = text.lower()
    return(text)

dh['Text'] = dh['Text'].apply(remove_pun)
X=dh.iloc[:,0]
Y=dh.iloc[:,1]
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=.2)
vec= feature_extraction.text.TfidfVectorizer(ngram_range=(1,2), analyzer='char')
model=pipeline.Pipeline([('vec',vec),('clf',linear_model.LogisticRegression())])
model.fit(x_train,y_train)
print(model.classes_)
pri=model.predict(x_test)
print(metrics.accuracy_score(y_test,pri)*100)
n=True
print(model.predict(['Hello there']))