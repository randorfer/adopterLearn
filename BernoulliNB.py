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
from sklearn.svm import LinearSVC 
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import chi2, f_classif
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA, NMF
from sklearn.pipeline import FeatureUnion
from sklearn.naive_bayes import BernoulliNB
train = pd.read_csv("data/train.csv")
score = pd.read_csv("data/score.csv")
score = score.fillna(method='ffill')
train = train.fillna(method='ffill')

X = train.values[:,3:]
y = train.values[:,2]
user_id = score.values[:,1]
X_score = score.values[:,2:]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5
)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('feature_select', SelectKBest(score_func=f_classif)),
    ('clf', BernoulliNB(binarize=1,fit_prior=True))
])
parameters = {
    'clf__alpha': (7,8,9,10,11),
    'feature_select__k': (8,9,10,11,12,13,14,15,16,17,20,24)
}

grid_search = GridSearchCV(
    pipeline,
    parameters,
    n_jobs=-1,
    verbose=1,
    cv=3,
    scoring='f1'
)

print("Performing grid search...")
print("pipeline:", [name for name, _ in pipeline.steps])
print("parameters:")
pprint(parameters)
t0 = time.time()
grid_search.fit(X,y)
print("done in %0.3fs" % (time.time() - t0))
print()

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

y_pred = grid_search.best_estimator_.predict(X_test)
precision, recall, fbeta_score, support = precision_recall_fscore_support(
    y_true=y_test, y_pred=y_pred, labels=1, average='binary'
)
print(fbeta_score)
pprint(confusion_matrix(y_test,y_pred))

predictions = grid_search.best_estimator_.predict(X_score)
timestr = time.strftime("%Y%m%d-%H%M%S") 
file_name = "output/{0}-{1}.csv".format("sklearn",timestr)
with open(file_name, 'w') as csvfile:
    csvfile.write("user_id,prediction(adopter)\n")
    i=0
    for prediction in predictions:
       csvfile.write("%s,%d\n" % (user_id[i],prediction))
       i += 1
