# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 13:18:29 2019

@author: Dan
"""

import scipy.stats
from sklearn.metrics import make_scorer
#from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

import sklearn_crfsuite

from utils import create_features, binary_f1
import pickle
import pandas as pd
from datetime import datetime

#import sklearn_crfsuite.metrics as metrics

DATA_PATH = "./data/{}.pkl"

print("{} : Reading train and test data sets...".format(datetime.now()))

with open(DATA_PATH.format("train"), "rb") as f:
    train = pickle.load(f)
    
with open(DATA_PATH.format("test"), "rb") as f:
    test = pickle.load(f)


crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    max_iterations=20,
    all_possible_transitions=True
)
params_space = {
    'c1': scipy.stats.expon(scale=0.5),
    'c2': scipy.stats.expon(scale=0.05),
}

X_train, y_train, X_test, y_test = create_features(train, test)

# use the same metric for evaluation
f1_scorer = make_scorer(binary_f1)
#f1_scorer = make_scorer(metrics.flat_f1_score,
#                        average='weighted', labels=["BEGIN", "IN", "OUT"])
# search
rs = RandomizedSearchCV(crf, params_space,
                        cv=3,
                        verbose=1,
                  #      n_jobs=-1,
                        n_iter=5,
                        scoring=f1_scorer)

print("{}: Performing Grid Search...".format(datetime.now()))
rs.fit(X_train, y_train)

results = pd.DataFrame(rs.cv_results_)

print("{}: Saving grid search results to file...".format(datetime.now()))
results.to_csv("./data/hyperparameter_tuning.csv", index=False)




