'''
INFO 523 Final Project DBSCAN code
https://www.kaggle.com/code/rpsuraj/outlier-detection-techniques-simplified
'''

import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

df = pd.read_csv("fetal_health.csv")
X = df[['abnormal_short_term_variability', 'baseline value']].values
db = DBSCAN(eps=3.0, min_samples=10).fit(X)

labels = db.labels_
#print(pd.Series(labels).value_counts())

plt.figure()
unique_labels = set(labels)
colors = ['blue', 'red']

for color, label in zip(colors, unique_labels):
    sample_mask = [True if l == label else False for l in labels]
    plt.plot(X[:,0][sample_mask], X[:, 1][sample_mask], 'o', color=color)

plt.xlabel('abnormal_short_term_variability')
plt.ylabel('baseline value')
plt.show()