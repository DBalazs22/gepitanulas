# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 00:30:55 2020

@author: Nagy Evelin
"""
from urllib.request import urlopen;
import numpy as np;
from numpy import array;
from sklearn.preprocessing import LabelBinarizer;
import pandas as pd;
from sklearn.model_selection import train_test_split; 
from sklearn.linear_model import LogisticRegression;
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, roc_curve, auc, plot_roc_curve, roc_auc_score;
from sklearn.naive_bayes import GaussianNB;

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data";
raw_data = urlopen(url);
data1 = np.genfromtxt(raw_data, delimiter=",", dtype='float64', names=True, encoding='utf-8');
pd.DataFrame(data1).head();
adat = [list(ele) for ele in data1];
data = array(adat);

x = data[:,1:8];
y = data[:,8];
print(x);
print(y);   

n = data.shape[0];
p = data.shape[1];

print(f'Number of records:{n}');
print(f'Number of attributes:{p}');

X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.25, random_state=2020, shuffle=True);

logreg_classifier = LogisticRegression(solver='liblinear');
logreg_classifier.fit(X_train,y_train);
score_train_logreg = logreg_classifier.score(X_train,y_train); 
score_test_logreg = logreg_classifier.score(X_test,y_test);
ypred_logreg = logreg_classifier.predict(X_test);
yprobab_logreg = logreg_classifier.predict_proba(X_test);
#cm_logreg_train = confusion_matrix(y_train, ypred_logreg);
cm_logreg_test = confusion_matrix(y_test, ypred_logreg);

naive_bayes_classifier = GaussianNB();
naive_bayes_classifier.fit(X_train,y_train);
score_train_naive_bayes = naive_bayes_classifier.score(X_train,y_train);
score_test_naive_bayes = naive_bayes_classifier.score(X_test,y_test);
ypred_naive_bayes = naive_bayes_classifier.predict(X_test);
yprobab_naive_bayes = naive_bayes_classifier.predict_proba(X_test);
cm_naive_bayes_test = confusion_matrix(y_test, ypred_naive_bayes);
#cm_naive_bayes_train = confusion_matrix(y_train, ypred_naive_bayes);

plot_confusion_matrix(naive_bayes_classifier, X_train, y_train); # NEM normalizalt
plot_confusion_matrix(naive_bayes_classifier, X_test, y_test);


def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)

print(multiclass_roc_auc_score(y_test, ypred_logreg));
