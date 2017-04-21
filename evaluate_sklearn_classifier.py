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
from sklearn.preprocessing import robust_scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_fscore_support, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import chi2
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


# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('feature_selection', SelectKBest()),
    ('clf', SVC())
])
parameters = {
    'feature_selection__k': (5, 10, 15, 20),
    'clf__kernel': ('linear','rbf')
}
'''
pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('clf', RandomForestClassifier(
                class_weight={1:.9,0:.1}, 
                max_depth=9
            ))
])
parameters = {
    'clf__max_features': ('sqrt','log2'),
    'clf__criterion': ('gini','entropy'),
    'clf__n_estimators': (5,10),
    'clf__oob_score': (True, False)
}
'''
grid_search = GridSearchCV(
    pipeline,
    parameters,
    n_jobs=-1,
    verbose=1,
    cv=5
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
file_name = "output/{0}-{1}.csv".format("results",timestr)
with open(file_name, 'w') as csvfile:
    csvfile.write("user_id,prediction(adopter)\n")
    i=0
    for prediction in predictions:
        csvfile.write("%s,%d\n" % (user_id[i],prediction))
        i += 1
