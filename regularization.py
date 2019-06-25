from sklearn_crfsuite import CRF

from utils import create_features, explain_weights, binarized_confusion_matrix
import pickle
from datetime import datetime
import os
import gensim.models.keyedvectors as word2vec

import numpy as np

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
    

X_train, y_train, X_test, y_test = create_features(train,
                                                   test,
                                                   dist=3,
                                                   include_PPMI=True,
                                                   word2vec=WORD2VEC)

c1_params = np.logspace(-1, 2, 4)
c2_params = np.logspace(-1, 2, 4)

for c2 in c2_params:
    for c1 in c1_params:
        crf = CRF(
            algorithm='lbfgs',
            c1=c1,
            c2=c2,
            max_iterations=100,
            all_possible_transitions=False,
        )
        print("{}: Training model...".format(datetime.now()))
        crf.fit(X_train, y_train)
        
        print("{}: Result for c1={} and c2={}".format(datetime.now(), c1, c2))
        
        weights_exp = explain_weights(crf, top=10)
        
        print(weights_exp["targets"][weights_exp["targets"]["target"]=="BEGIN"])
        
        preds = crf.predict(X_test)
        binarized_confusion_matrix(preds, y_test)

