# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 14:00:12 2020

Task: K-means clustering of Abalone dataframe
Input: Abalone dataframe
Output: cluster statistics, DB score, cluster plot

@author: Nagy Evelin
"""

import numpy as np;
import pandas as pd;
from sklearn.cluster import KMeans;
from sklearn.decomposition import PCA;
from matplotlib import pyplot as plt;
from sklearn.metrics import davies_bouldin_score;

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data', 
            sep=",", names = ['Sex','Length','Diameter','Height','Whole weight','Shucked weight',
                               'Viscera weight', 'Shell weight', 'Rings']);

adat = df.get(['Sex','Length','Diameter','Height','Whole weight','Shucked weight',
                               'Viscera weight', 'Shell weight', 'Rings']);

n = adat.shape[0];
p = adat.shape[1];
abalone_by_sex = adat.groupby(by='Sex');

print(f'Number of records:{n}');
print(f'Number of attributes:{p}');

mean = abalone_by_sex.mean();
std = abalone_by_sex.std();
corr = abalone_by_sex.corr();
desc_stat = abalone_by_sex.describe();
#print(mean.mean());

"""
plt.figure(1);
pd.plotting.andrews_curves(adat, class_column='Sex',color=['blue','green','red']);
plt.show();
# Parallel axis
plt.figure(2);
pd.plotting.parallel_coordinates(adat,class_column='Sex',color=['blue','green','red']);
plt.show();
"""

x = adat[['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight']];
y = adat['Rings'];

clusters = 2;

kmeans = KMeans(n_clusters=clusters, random_state=2020);
kmeans.fit(x);
labels = kmeans.labels_;
centers = kmeans.cluster_centers_;
sse = kmeans.inertia_;
score = kmeans.score(x);

DB = davies_bouldin_score(x, labels);

print(f'Within SSE: {sse}');
print(f'Davies-Bouldin index: {DB}');

pca = PCA(n_components=2);
pca.fit(x);
adatok_pc = pca.transform(x);
centers_pc = pca.transform(centers);

fig = plt.figure(1);
plt.title('Clustering');
plt.xlabel('PC1');
plt.ylabel('PC2');
plt.scatter(adatok_pc[:,0],adatok_pc[:,1],s=50,c=labels);
plt.scatter(centers_pc[:,0],centers_pc[:,1],s=200,c='red',marker='X');
#plt.legend();
plt.show();

Max_K = 30;
SSE = np.zeros((Max_K-2));
DB = np.zeros((Max_K-2));
for i in range(Max_K-2):
    n_c = i+2;
    kmeans = KMeans(n_clusters=n_c, random_state=2020);
    kmeans.fit(x);
    Klabels = kmeans.labels_;
    SSE[i] = kmeans.inertia_;
    DB[i] = davies_bouldin_score(x, Klabels);
   
fig = plt.figure(3);
plt.title('Sum of squares of error curve');
plt.xlabel('Number of clusters');
plt.ylabel('SSE');
plt.plot(np.arange(2,Max_K),SSE, color='red')
plt.show();

fig = plt.figure(4);
plt.title('Davies-Bouldin score curve');
plt.xlabel('Number of clusters');
plt.ylabel('DB index');
plt.plot(np.arange(2,Max_K),DB, color='blue')
plt.show();