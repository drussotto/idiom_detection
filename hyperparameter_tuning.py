import scipy.stats
from sklearn.metrics import make_scorer
#from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

import sklearn_crfsuite

from utils import create_features, binary_f1
import pickle
import pandas as pd
from datetime import datetime
import os
import gensim.models.keyedvectors as word2vec

DATA_PATH = "./data/{}.pkl"

print("{}: Loading pretrained word2vec model")
if os.path.exists("./data/GoogleNews-vectors-negative300.bin"):
    WORD2VEC = word2vec.KeyedVectors.load_word2vec_format(
        "./data/GoogleNews-vectors-negative300.bin",
        binary=True)
else:
    print("Pretrain Word2Vec model not found!")


print("{} : Reading train and test data sets...".format(datetime.now()))
with open(DATA_PATH.format("train"), "rb") as f:
    train = pickle.load(f)
    
with open(DATA_PATH.format("test"), "rb") as f:
    test = pickle.load(f)


crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    max_iterations=100,
    all_possible_transitions=True
)
params_space = {
    'c1': scipy.stats.expon(scale=0.5),
    'c2': scipy.stats.expon(scale=0.05),
}

OUT_PATH = "./data/hyperparameter_tuning_{}.csv"

f1_scorer = make_scorer(binary_f1)

rs = RandomizedSearchCV(crf, params_space,
                        cv=3,
                        n_iter=20,
                        scoring=f1_scorer)

X_train, y_train, X_test, y_test = create_features(train,
                                                   test,
                                                   dist=3,
                                                   include_PPMI=True,
                                                   word2vec=WORD2VEC)



print("{}: Performing Grid Search with Word2Vec 3 words ahead/behind..."\
      .format(datetime.now()))

rs.fit(X_train, y_train)

results = pd.DataFrame(rs.cv_results_)

print("{}: Saving grid search results to file...".format(datetime.now()))
results.to_csv(OUT_PATH.format("PPMI_W2V_3"), index=False)




