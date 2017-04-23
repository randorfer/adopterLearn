#%%
from __future__ import print_function
import pandas as pd
import numpy as np
import time
from pprint import pprint
import logging

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import robust_scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_fscore_support, classification_report
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import chi2
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA, NMF
from sklearn.pipeline import FeatureUnion
train = pd.read_csv("data/train.csv")
score = pd.read_csv("data/score.csv")
score = score.fillna(method='ffill')
train = train.fillna(method='ffill')

X = train.values[:,3:]
y = train.values[:,2]
user_id = score.values[:,1]
X_score = score.values[:,2:]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3
)

from tpot import TPOTClassifier

pipeline_optimizer = TPOTClassifier(generations=8, population_size=50, cv=5, n_jobs=6,
                                    random_state=42, verbosity=3, scoring='f1')
pipeline_optimizer.fit(X, y)
print(pipeline_optimizer.score(X_test, y_test))
pipeline_optimizer.export('tpot_exported_pipeline.py')

y_pred = pipeline_optimizer.predict(X_test)
precision, recall, fbeta_score, support = precision_recall_fscore_support(
    y_true=y_test, y_pred=y_pred, labels=1, average='binary'
)
print(fbeta_score)
pprint(confusion_matrix(y_test,y_pred))

predictions = pipeline_optimizer.predict(X_score)
timestr = time.strftime("%Y%m%d-%H%M%S") 
file_name = "output/{0}-{1}.csv".format("pipeline",timestr)
with open(file_name, 'w') as csvfile:
    csvfile.write("user_id,prediction(adopter)\n")
    i=0
    for prediction in predictions:
       csvfile.write("%s,%d\n" % (user_id[i],prediction))
       i += 1
