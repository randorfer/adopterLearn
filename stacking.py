#%%
from __future__ import print_function
import pandas as pd
import numpy as np
import time
from pprint import pprint
import logging

from sklearn import model_selection
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_fscore_support, classification_report
from sklearn.metrics import f1_score

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 12, 4

__target__ = 'adopter'
__id__ = 'user_id'
train = pd.read_csv("data/train.csv")
score = pd.read_csv("data/score.csv")
def modelfit(alg, dtrain, score, predictors, target, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target],eval_metric=f1_score)
        
    #Predict training set:
    score_predictions = alg.predict(score[predictors])
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    print("\nModel Report")
    _fscore_ = f1_score(dtrain[target].values, dtrain_predictions)
    print("F1: {0}".format(_fscore_))
    pprint(confusion_matrix(dtrain[target].values, dtrain_predictions))

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    return score_predictions
predictors = [x for x in train.columns if x not in [__target__, __id__, 'row_number']]

#%%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB 
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier
clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = BernoulliNB(alpha=10, binarize=True)
xgb1 = XGBClassifier(
    learning_rate =0.01,
    n_estimators=5000,
    max_depth=2,
    min_child_weight=6,
    gamma=0,
    subsample=0.6,
    objective= 'binary:logistic',
    nthread=4,
    scale_pos_weight=15,
    reg_alpha=0.0001,
    reg_lambda=1,
    seed=27
)
sclf = StackingClassifier(classifiers=[clf3],
                          meta_classifier=xgb1)
param_test1 = {'bernoullinb__alpha': [10,11]}

gsearch1 = GridSearchCV(estimator=sclf, 
                    param_grid=param_test1, 
                    cv=5,
                    refit=True,
                    scoring='f1')
gsearch1.fit(train[predictors],train[__target__])
pprint(gsearch1.grid_scores_)
pprint(gsearch1.best_params_)
pprint(gsearch1.best_score_)
#%%
xgb1 = XGBClassifier(
 learning_rate =0.01,
 n_estimators=5000,
 max_depth=2,
 min_child_weight=6,
 gamma=0,
 subsample=0.6,
 colsample_bytree=0.9,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=15,
 reg_alpha=0.0001,
 reg_lambda=1,
 seed=27)
score_predictions = modelfit(xgb1, train, score, predictors, __target__)
#%%
timestr = time.strftime("%Y%m%d-%H%M%S") 
file_name = "output/{0}-{1}.csv".format("xgboost",timestr)
user_id = score.values[:,1]
with open(file_name, 'w') as csvfile:
    csvfile.write("user_id,prediction(adopter)\n")
    i=0
    for prediction in score_predictions:
       csvfile.write("%s,%d\n" % (user_id[i],prediction))
       i += 1