# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 00:30:55 2020

@author: Nagy Evelin
"""

import numpy as np;
import pandas as pd;
import seaborn as sns;
from matplotlib import pyplot as plt;
import matplotlib.colors as col;
from sklearn.model_selection import train_test_split; 
from sklearn.linear_model import LinearRegression;
from sklearn.linear_model import LogisticRegression;
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, roc_curve, auc, plot_roc_curve;
from sklearn.naive_bayes import GaussianNB;

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data', 
            sep=",", names = ['Sex','Length','Diameter','Height','Whole weight','Shucked weight',
                               'Viscera weight', 'Shell weight', 'Rings']);

adat = df.get(['Sex','Length','Diameter','Height','Whole weight','Shucked weight',
                               'Viscera weight', 'Shell weight', 'Rings']);
n = adat.shape[0];
p = adat.shape[1];

print(f'Number of records:{n}');
print(f'Number of attributes:{p}');

adat ['Rings'] = adat ['Rings'].astype(float);
teszt = adat[['Rings']];

x = adat[['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight']];
y = adat['Rings'];

X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.25, random_state=2020, shuffle=True);

#print(type(adat));
#print(adat.keys());

sns.pairplot(adat);

reg = LinearRegression();
reg.fit(X_train,y_train);
print("Coef:",reg.coef_);
print("Intercept:",reg.intercept_);
#pd.DataFrame(reg.coef_, x.columns, columns = ['Coeff']);
predictions = reg.predict(X_test);
plt.scatter(y_test, predictions);
#plt.plot([50,350],[50,350],color='red');
#plt.show();

intercept = reg.intercept_;
coef = reg.coef_;
score_train = reg.score(X_train,y_train);
score_test = reg.score(X_test,y_test);
y_test_pred = reg.predict(X_test);
print("Test score:", score_test);
print("Train score:",score_train);

logreg_classifier = LogisticRegression(solver='liblinear');
logreg_classifier.fit(X_train,y_train);
intercept1 = logreg_classifier.intercept_[0];
weight = logreg_classifier.coef_[0,:];

score_train = logreg_classifier.score(X_train,y_train);
score_test = logreg_classifier.score(X_test,y_test);

ypred_logreg = logreg_classifier.predict(X_train);
accuracy_logreg_train = logreg_classifier.score(X_train,y_train);
cm_logreg_train = confusion_matrix(y_train, ypred_logreg); 
ypred_logreg = logreg_classifier.predict(X_test);
cm_logreg_test = confusion_matrix(y_test, ypred_logreg); 
yprobab_logreg = logreg_classifier.predict_proba(X_test); 
accuracy_logreg_test = logreg_classifier.score(X_test,y_test);

#plot_confusion_matrix(logreg_classifier, X_train, y_train);

#plot_confusion_matrix(logreg_classifier, X_test, y_test);
naive_bayes_classifier = GaussianNB();
naive_bayes_classifier.fit(X_train,y_train);
ypred_naive_bayes = naive_bayes_classifier.predict(X_train);
cm_naive_bayes_train = confusion_matrix(y_train, ypred_naive_bayes); 
ypred_naive_bayes = naive_bayes_classifier.predict(X_test);  
cm_naive_bayes_test = confusion_matrix(y_test, ypred_naive_bayes);
yprobab_naive_bayes = naive_bayes_classifier.predict_proba(X_test);

plot_confusion_matrix(naive_bayes_classifier, X_train, y_train, display_labels = "Teszt");

plot_confusion_matrix(naive_bayes_classifier, X_test, y_test, display_labels = "Teszt1");

plot_roc_curve(logreg_classifier, X_test, y_test);
plot_roc_curve(naive_bayes_classifier, X_test, y_test);

fpr_logreg, tpr_logreg, _ = roc_curve(y_test, yprobab_logreg[:,1]);
roc_auc_logreg = auc(fpr_logreg, tpr_logreg);

fpr_naive_bayes, tpr_naive_bayes, _ = roc_curve(y_test, yprobab_naive_bayes[:,1]);
roc_auc_naive_bayes = auc(fpr_naive_bayes, tpr_naive_bayes);

plt.figure(7);
lw = 2;
plt.plot(fpr_logreg, tpr_logreg, color='red',
         lw=lw, label='Logistic regression (AUC = %0.2f)' % roc_auc_logreg);
plt.plot(fpr_naive_bayes, tpr_naive_bayes, color='blue',
         lw=lw, label='Naive Bayes (AUC = %0.2f)' % roc_auc_naive_bayes);
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--');
plt.xlim([0.0, 1.0]);
plt.ylim([0.0, 1.05]);
plt.xlabel('False Positive Rate');
plt.ylabel('True Positive Rate');
plt.title('Receiver operating characteristic curve');
plt.legend(loc="lower right");
plt.show();