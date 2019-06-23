import pickle
from utils import sent2features, sent2labels, draw_cm
from sklearn_crfsuite import CRF
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import eli5
from datetime import datetime
import os
import gensim.models.keyedvectors as word2vec

DATA_PATH = "./data/{}.pkl"

with open(DATA_PATH.format("train"), "rb") as f:
    train = pickle.load(f)
    
with open(DATA_PATH.format("test"), "rb") as f:
    test = pickle.load(f)
    

# =============================================================================
# BASELINE    
# =============================================================================
    
X_train = [sent2features(s) for s in train]
y_train = [sent2labels(s) for s in train]

X_test = [sent2features(s) for s in test]
y_test = [sent2labels(s) for s in test]
    
    
crf = CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=20,
    all_possible_transitions=False,
)

crf.fit(X_train, y_train)


predictions = crf.predict(X_test)

report = classification_report(MultiLabelBinarizer().fit_transform(y_test),
                               MultiLabelBinarizer().fit_transform(predictions),
                               target_names=["BEGIN", "IN", "OUT"],
                               digits=3)

print(report)

#mtx = multilabel_confusion_matrix(MultiLabelBinarizer().fit_transform(y_test),
#                                  MultiLabelBinarizer().fit_transform(predictions))
#
#
#print(mtx)



predictions_bin = ["idiom" if p[0] else "no_idiom" \
                   for p in MultiLabelBinarizer().fit_transform(predictions)]
y_test_bin = ["idiom" if y[0] else "no_idiom" \
              for y in MultiLabelBinarizer().fit_transform(y_test)]

draw_cm(y_test_bin, predictions_bin)

weights_exp = eli5.formatters.explain_weights_dfs(crf, top=30)
weights_exp["targets"][weights_exp["targets"]["target"]=="BEGIN"]
weights_exp["targets"][weights_exp["targets"]["target"]=="IN"]
weights_exp["targets"][weights_exp["targets"]["target"]=="OUT"]



# =============================================================================
# PMI
# =============================================================================

print("{}: Creating features for train set...".format(datetime.now()))
X_train = [sent2features(s, include_PMI=True) for s in train]
print("{}: Getting labels for train set...".format(datetime.now()))
y_train = [sent2labels(s) for s in train]

print("{}: Creating features for test set".format(datetime.now()))
X_test = [sent2features(s, include_PMI=True) for s in test]
print("{}: Getting labels for test set...".format(datetime.now()))
y_test = [sent2labels(s) for s in test]
print("{}: Finished!".format(datetime.now()))

crf = CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=20,
    all_possible_transitions=False,
)


crf.fit(X_train, y_train)

predictions = crf.predict(X_test)



report = classification_report(MultiLabelBinarizer().fit_transform(y_test),
                               MultiLabelBinarizer(classes=["BEGIN", "IN", "OUT"]).fit_transform(predictions),
                               target_names=["BEGIN", "IN", "OUT"],
                               digits=3)

print(report)


predictions_bin = ["idiom" if p[0] else "no_idiom" \
                   for p in MultiLabelBinarizer(classes=["BEGIN", "IN", "OUT"]).fit_transform(predictions)]
y_test_bin = ["idiom" if y[0] else "no_idiom" \
              for y in MultiLabelBinarizer().fit_transform(y_test)]

draw_cm(y_test_bin, predictions_bin)


weights_exp = eli5.formatters.explain_weights_dfs(crf, top=30)
weights_exp["targets"][weights_exp["targets"]["target"]=="BEGIN"]
weights_exp["targets"][weights_exp["targets"]["target"]=="IN"]
weights_exp["targets"][weights_exp["targets"]["target"]=="OUT"]


# =============================================================================
# Postive PMI
# =============================================================================

print("{}: Creating features for train set...".format(datetime.now()))
X_train = [sent2features(s, include_PPMI=True) for s in train]
print("{}: Getting labels for train set...".format(datetime.now()))
y_train = [sent2labels(s) for s in train]

print("{}: Creating features for test set".format(datetime.now()))
X_test = [sent2features(s, include_PPMI=True) for s in test]
print("{}: Getting labels for test set...".format(datetime.now()))
y_test = [sent2labels(s) for s in test]
print("{}: Finished!".format(datetime.now()))

crf = CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=20,
    all_possible_transitions=False,
)


crf.fit(X_train, y_train)

predictions = crf.predict(X_test)



report = classification_report(MultiLabelBinarizer().fit_transform(y_test),
                               MultiLabelBinarizer(classes=["BEGIN", "IN", "OUT"]).fit_transform(predictions),
                               target_names=["BEGIN", "IN", "OUT"],
                               digits=3)

print(report)


predictions_bin = ["idiom" if p[0] else "no_idiom" \
                   for p in MultiLabelBinarizer(classes=["BEGIN", "IN", "OUT"]).fit_transform(predictions)]
y_test_bin = ["idiom" if y[0] else "no_idiom" \
              for y in MultiLabelBinarizer().fit_transform(y_test)]

draw_cm(y_test_bin, predictions_bin)


weights_exp = eli5.formatters.explain_weights_dfs(crf, top=30)
weights_exp["targets"][weights_exp["targets"]["target"]=="BEGIN"]
weights_exp["targets"][weights_exp["targets"]["target"]=="IN"]
weights_exp["targets"][weights_exp["targets"]["target"]=="OUT"]


# =============================================================================
# WORD2VEC
# =============================================================================


## This is sensitive to capitalization - lowercase (?)
if os.path.exists("./data/GoogleNews-vectors-negative300.bin"):
    WORD2VEC = word2vec.KeyedVectors.load_word2vec_format(
        "./data/GoogleNews-vectors-negative300.bin",
        binary=True)


print("{}: Creating features for train set...".format(datetime.now()))
X_train = [sent2features(s, word2vec=WORD2VEC) for s in train]
print("{}: Getting labels for train set...".format(datetime.now()))
y_train = [sent2labels(s) for s in train]

print("{}: Creating features for test set".format(datetime.now()))
X_test = [sent2features(s, word2vec=WORD2VEC) for s in test]
print("{}: Getting labels for test set...".format(datetime.now()))
y_test = [sent2labels(s) for s in test]
print("{}: Finished!".format(datetime.now()))

crf = CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=20,
    all_possible_transitions=False,
)


crf.fit(X_train, y_train)

predictions = crf.predict(X_test)



report = classification_report(MultiLabelBinarizer().fit_transform(y_test),
                               MultiLabelBinarizer(classes=["BEGIN", "IN", "OUT"]).fit_transform(predictions),
                               target_names=["BEGIN", "IN", "OUT"],
                               digits=3)

print(report)


predictions_bin = ["idiom" if p[0] else "no_idiom" \
                   for p in MultiLabelBinarizer(classes=["BEGIN", "IN", "OUT"]).fit_transform(predictions)]
y_test_bin = ["idiom" if y[0] else "no_idiom" \
              for y in MultiLabelBinarizer().fit_transform(y_test)]

draw_cm(y_test_bin, predictions_bin)



# =============================================================================
# BASIC FEATURES 2 AHEAD
# =============================================================================


print("{}: Creating features for train set...".format(datetime.now()))
X_train = [sent2features(s, dist=2) for s in train]
print("{}: Getting labels for train set...".format(datetime.now()))
y_train = [sent2labels(s) for s in train]

print("{}: Creating features for test set".format(datetime.now()))
X_test = [sent2features(s, dist=2) for s in test]
print("{}: Getting labels for test set...".format(datetime.now()))
y_test = [sent2labels(s) for s in test]
print("{}: Finished!".format(datetime.now()))
    
    
crf = CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=20,
    all_possible_transitions=False,
)

crf.fit(X_train, y_train)


predictions = crf.predict(X_test)

report = classification_report(MultiLabelBinarizer().fit_transform(y_test),
                               MultiLabelBinarizer().fit_transform(predictions),
                               target_names=["BEGIN", "IN", "OUT"],
                               digits=3)

print(report)

predictions_bin = ["idiom" if p[0] else "no_idiom" \
                   for p in MultiLabelBinarizer().fit_transform(predictions)]
y_test_bin = ["idiom" if y[0] else "no_idiom" \
              for y in MultiLabelBinarizer().fit_transform(y_test)]

draw_cm(y_test_bin, predictions_bin)


# =============================================================================
# BASIC FEATURES 3 AHEAD
# =============================================================================


print("{}: Creating features for train set...".format(datetime.now()))
X_train = [sent2features(s, dist=3) for s in train]
print("{}: Getting labels for train set...".format(datetime.now()))
y_train = [sent2labels(s) for s in train]

print("{}: Creating features for test set".format(datetime.now()))
X_test = [sent2features(s, dist=3) for s in test]
print("{}: Getting labels for test set...".format(datetime.now()))
y_test = [sent2labels(s) for s in test]
print("{}: Finished!".format(datetime.now()))
    
    
crf = CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=20,
    all_possible_transitions=False,
)

crf.fit(X_train, y_train)


predictions = crf.predict(X_test)

report = classification_report(MultiLabelBinarizer().fit_transform(y_test),
                               MultiLabelBinarizer().fit_transform(predictions),
                               target_names=["BEGIN", "IN", "OUT"],
                               digits=3)

print(report)

predictions_bin = ["idiom" if p[0] else "no_idiom" \
                   for p in MultiLabelBinarizer().fit_transform(predictions)]
y_test_bin = ["idiom" if y[0] else "no_idiom" \
              for y in MultiLabelBinarizer().fit_transform(y_test)]

draw_cm(y_test_bin, predictions_bin)


# =============================================================================
# PPMI 2 AHEAD
# =============================================================================


print("{}: Creating features for train set...".format(datetime.now()))
X_train = [sent2features(s, dist=2, include_PPMI=True) for s in train]
print("{}: Getting labels for train set...".format(datetime.now()))
y_train = [sent2labels(s) for s in train]

print("{}: Creating features for test set".format(datetime.now()))
X_test = [sent2features(s, dist=2, include_PPMI=True) for s in test]
print("{}: Getting labels for test set...".format(datetime.now()))
y_test = [sent2labels(s) for s in test]
print("{}: Finished!".format(datetime.now()))
    
    
crf = CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=20,
    all_possible_transitions=False,
)

crf.fit(X_train, y_train)


predictions = crf.predict(X_test)

report = classification_report(MultiLabelBinarizer().fit_transform(y_test),
                               MultiLabelBinarizer().fit_transform(predictions),
                               target_names=["BEGIN", "IN", "OUT"],
                               digits=3)

print(report)

predictions_bin = ["idiom" if p[0] else "no_idiom" \
                   for p in MultiLabelBinarizer().fit_transform(predictions)]
y_test_bin = ["idiom" if y[0] else "no_idiom" \
              for y in MultiLabelBinarizer().fit_transform(y_test)]

draw_cm(y_test_bin, predictions_bin)


# =============================================================================
# WORD2VEC 2 AHEAD
# =============================================================================


print("{}: Creating features for train set...".format(datetime.now()))
X_train = [sent2features(s, dist=2, word2vec=WORD2VEC) for s in train]
print("{}: Getting labels for train set...".format(datetime.now()))
y_train = [sent2labels(s) for s in train]

print("{}: Creating features for test set".format(datetime.now()))
X_test = [sent2features(s, dist=2, word2vec=WORD2VEC) for s in test]
print("{}: Getting labels for test set...".format(datetime.now()))
y_test = [sent2labels(s) for s in test]
print("{}: Finished!".format(datetime.now()))
    
    
crf = CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=20,
    all_possible_transitions=False,
)

crf.fit(X_train, y_train)


predictions = crf.predict(X_test)

report = classification_report(MultiLabelBinarizer().fit_transform(y_test),
                               MultiLabelBinarizer().fit_transform(predictions),
                               target_names=["BEGIN", "IN", "OUT"],
                               digits=3)

print(report)

predictions_bin = ["idiom" if p[0] else "no_idiom" \
                   for p in MultiLabelBinarizer().fit_transform(predictions)]
y_test_bin = ["idiom" if y[0] else "no_idiom" \
              for y in MultiLabelBinarizer().fit_transform(y_test)]

draw_cm(y_test_bin, predictions_bin)


# =============================================================================
# PPMI 3 AHEAD
# =============================================================================


print("{}: Creating features for train set...".format(datetime.now()))
X_train = [sent2features(s, dist=3, include_PPMI=True) for s in train]
print("{}: Getting labels for train set...".format(datetime.now()))
y_train = [sent2labels(s) for s in train]

print("{}: Creating features for test set".format(datetime.now()))
X_test = [sent2features(s, dist=3, include_PPMI=True) for s in test]
print("{}: Getting labels for test set...".format(datetime.now()))
y_test = [sent2labels(s) for s in test]
print("{}: Finished!".format(datetime.now()))
    
    
crf = CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=20,
    all_possible_transitions=False,
)

crf.fit(X_train, y_train)


predictions = crf.predict(X_test)

report = classification_report(MultiLabelBinarizer().fit_transform(y_test),
                               MultiLabelBinarizer().fit_transform(predictions),
                               target_names=["BEGIN", "IN", "OUT"],
                               digits=3)

print(report)

predictions_bin = ["idiom" if p[0] else "no_idiom" \
                   for p in MultiLabelBinarizer().fit_transform(predictions)]
y_test_bin = ["idiom" if y[0] else "no_idiom" \
              for y in MultiLabelBinarizer().fit_transform(y_test)]

draw_cm(y_test_bin, predictions_bin)


# =============================================================================
# WORD2VEC 3 AHEAD
# =============================================================================


print("{}: Creating features for train set...".format(datetime.now()))
X_train = [sent2features(s, dist=3, word2vec=WORD2VEC) for s in train]
print("{}: Getting labels for train set...".format(datetime.now()))
y_train = [sent2labels(s) for s in train]

print("{}: Creating features for test set".format(datetime.now()))
X_test = [sent2features(s, dist=3, word2vec=WORD2VEC) for s in test]
print("{}: Getting labels for test set...".format(datetime.now()))
y_test = [sent2labels(s) for s in test]
print("{}: Finished!".format(datetime.now()))
    
    
crf = CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=20,
    all_possible_transitions=False,
)

crf.fit(X_train, y_train)


predictions = crf.predict(X_test)

report = classification_report(MultiLabelBinarizer().fit_transform(y_test),
                               MultiLabelBinarizer().fit_transform(predictions),
                               target_names=["BEGIN", "IN", "OUT"],
                               digits=3)

print(report)

predictions_bin = ["idiom" if p[0] else "no_idiom" \
                   for p in MultiLabelBinarizer().fit_transform(predictions)]
y_test_bin = ["idiom" if y[0] else "no_idiom" \
              for y in MultiLabelBinarizer().fit_transform(y_test)]

draw_cm(y_test_bin, predictions_bin)


# =============================================================================
# WORD2VEC AND PPMI
# =============================================================================

print("{}: Creating features for train set...".format(datetime.now()))
X_train = [sent2features(s, include_PPMI=True, word2vec=WORD2VEC) for s in train]
print("{}: Getting labels for train set...".format(datetime.now()))
y_train = [sent2labels(s) for s in train]

print("{}: Creating features for test set".format(datetime.now()))
X_test = [sent2features(s, include_PPMI=True, word2vec=WORD2VEC) for s in test]
print("{}: Getting labels for test set...".format(datetime.now()))
y_test = [sent2labels(s) for s in test]
print("{}: Finished!".format(datetime.now()))

crf = CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=20,
    all_possible_transitions=False,
)


crf.fit(X_train, y_train)

predictions = crf.predict(X_test)

report = classification_report(MultiLabelBinarizer().fit_transform(y_test),
                               MultiLabelBinarizer(classes=["BEGIN", "IN", "OUT"]).fit_transform(predictions),
                               target_names=["BEGIN", "IN", "OUT"],
                               digits=3)

print(report)


predictions_bin = ["idiom" if p[0] else "no_idiom" \
                   for p in MultiLabelBinarizer(classes=["BEGIN", "IN", "OUT"]).fit_transform(predictions)]
y_test_bin = ["idiom" if y[0] else "no_idiom" \
              for y in MultiLabelBinarizer().fit_transform(y_test)]

draw_cm(y_test_bin, predictions_bin)

# =============================================================================
# WORD2VEC AND PPMI 2 AHEAD
# =============================================================================

print("{}: Creating features for train set...".format(datetime.now()))
X_train = [sent2features(s, dist=2, include_PPMI=True, word2vec=WORD2VEC) for s in train]
print("{}: Getting labels for train set...".format(datetime.now()))
y_train = [sent2labels(s) for s in train]

print("{}: Creating features for test set".format(datetime.now()))
X_test = [sent2features(s, dist=2, include_PPMI=True, word2vec=WORD2VEC) for s in test]
print("{}: Getting labels for test set...".format(datetime.now()))
y_test = [sent2labels(s) for s in test]
print("{}: Finished!".format(datetime.now()))

crf = CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=20,
    all_possible_transitions=False,
)


crf.fit(X_train, y_train)

predictions = crf.predict(X_test)

report = classification_report(MultiLabelBinarizer().fit_transform(y_test),
                               MultiLabelBinarizer(classes=["BEGIN", "IN", "OUT"]).fit_transform(predictions),
                               target_names=["BEGIN", "IN", "OUT"],
                               digits=3)

print(report)


predictions_bin = ["idiom" if p[0] else "no_idiom" \
                   for p in MultiLabelBinarizer(classes=["BEGIN", "IN", "OUT"]).fit_transform(predictions)]
y_test_bin = ["idiom" if y[0] else "no_idiom" \
              for y in MultiLabelBinarizer().fit_transform(y_test)]

draw_cm(y_test_bin, predictions_bin)


# =============================================================================
# WORD2VEC AND PPMI 3 AHEAD
# =============================================================================

print("{}: Creating features for train set...".format(datetime.now()))
X_train = [sent2features(s, dist=3, include_PPMI=True, word2vec=WORD2VEC) for s in train]
print("{}: Getting labels for train set...".format(datetime.now()))
y_train = [sent2labels(s) for s in train]

print("{}: Creating features for test set".format(datetime.now()))
X_test = [sent2features(s, dist=3, include_PPMI=True, word2vec=WORD2VEC) for s in test]
print("{}: Getting labels for test set...".format(datetime.now()))
y_test = [sent2labels(s) for s in test]
print("{}: Finished!".format(datetime.now()))

crf = CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=20,
    all_possible_transitions=False,
)


crf.fit(X_train, y_train)

predictions = crf.predict(X_test)

report = classification_report(MultiLabelBinarizer().fit_transform(y_test),
                               MultiLabelBinarizer(classes=["BEGIN", "IN", "OUT"]).fit_transform(predictions),
                               target_names=["BEGIN", "IN", "OUT"],
                               digits=3)

print(report)


predictions_bin = ["idiom" if p[0] else "no_idiom" \
                   for p in MultiLabelBinarizer(classes=["BEGIN", "IN", "OUT"]).fit_transform(predictions)]
y_test_bin = ["idiom" if y[0] else "no_idiom" \
              for y in MultiLabelBinarizer().fit_transform(y_test)]

draw_cm(y_test_bin, predictions_bin)




# =============================================================================
# BASELINE REGULARIZED
# =============================================================================

X_train = [sent2features(s) for s in train]
y_train = [sent2labels(s) for s in train]

X_test = [sent2features(s) for s in test]
y_test = [sent2labels(s) for s in test]


crf = CRF(
    algorithm='lbfgs',
    c1=200,
    c2=0.1,
    max_iterations=20,
    all_possible_transitions=False,
)

crf.fit(X_train, y_train)

predictions = crf.predict(X_test)



report = classification_report(MultiLabelBinarizer().fit_transform(y_test),
                               MultiLabelBinarizer(classes=["BEGIN", "IN", "OUT"]).fit_transform(predictions),
                               target_names=["BEGIN", "IN", "OUT"],
                               digits=3)

print(report)


predictions_bin = ["idiom" if p[0] else "no_idiom" \
                   for p in MultiLabelBinarizer(classes=["BEGIN", "IN", "OUT"]).fit_transform(predictions)]
y_test_bin = ["idiom" if y[0] else "no_idiom" \
              for y in MultiLabelBinarizer().fit_transform(y_test)]

draw_cm(y_test_bin, predictions_bin)


weights_exp = eli5.formatters.explain_weights_dfs(crf, top=30)
weights_exp["targets"][weights_exp["targets"]["target"]=="BEGIN"]
weights_exp["targets"][weights_exp["targets"]["target"]=="IN"]
weights_exp["targets"][weights_exp["targets"]["target"]=="OUT"]









# =============================================================================
# PMI REGULARIZED
# =============================================================================

print("{}: Creating features for train set...".format(datetime.now()))
X_train = [sent2features(s, include_PMI=True) for s in train]
print("{}: Getting labels for train set...".format(datetime.now()))
y_train = [sent2labels(s) for s in train]

print("{}: Creating features for test set".format(datetime.now()))
X_test = [sent2features(s, include_PMI=True) for s in test]
print("{}: Getting labels for test set...".format(datetime.now()))
y_test = [sent2labels(s) for s in test]
print("{}: Finished!".format(datetime.now()))

crf = CRF(
    algorithm='lbfgs',
    c1=200,
    c2=0.1,
    max_iterations=20,
    all_possible_transitions=False,
)


crf.fit(X_train, y_train)

predictions = crf.predict(X_test)



report = classification_report(MultiLabelBinarizer().fit_transform(y_test),
                               MultiLabelBinarizer(classes=["BEGIN", "IN", "OUT"]).fit_transform(predictions),
                               target_names=["BEGIN", "IN", "OUT"],
                               digits=3)

print(report)


predictions_bin = ["idiom" if p[0] else "no_idiom" \
                   for p in MultiLabelBinarizer(classes=["BEGIN", "IN", "OUT"]).fit_transform(predictions)]
y_test_bin = ["idiom" if y[0] else "no_idiom" \
              for y in MultiLabelBinarizer().fit_transform(y_test)]

draw_cm(y_test_bin, predictions_bin)


weights_exp = eli5.formatters.explain_weights_dfs(crf, top=30)
weights_exp["targets"][weights_exp["targets"]["target"]=="BEGIN"]
weights_exp["targets"][weights_exp["targets"]["target"]=="IN"]
weights_exp["targets"][weights_exp["targets"]["target"]=="OUT"]


# =============================================================================
# Postive PMI REGULARIZED
# =============================================================================

print("{}: Creating features for train set...".format(datetime.now()))
X_train = [sent2features(s, include_PPMI=True) for s in train]
print("{}: Getting labels for train set...".format(datetime.now()))
y_train = [sent2labels(s) for s in train]

print("{}: Creating features for test set".format(datetime.now()))
X_test = [sent2features(s, include_PPMI=True) for s in test]
print("{}: Getting labels for test set...".format(datetime.now()))
y_test = [sent2labels(s) for s in test]
print("{}: Finished!".format(datetime.now()))

crf = CRF(
    algorithm='lbfgs',
    c1=200,
    c2=0.1,
    max_iterations=20,
    all_possible_transitions=False,
)


crf.fit(X_train, y_train)

predictions = crf.predict(X_test)



report = classification_report(MultiLabelBinarizer().fit_transform(y_test),
                               MultiLabelBinarizer(classes=["BEGIN", "IN", "OUT"]).fit_transform(predictions),
                               target_names=["BEGIN", "IN", "OUT"],
                               digits=3)

print(report)


predictions_bin = ["idiom" if p[0] else "no_idiom" \
                   for p in MultiLabelBinarizer(classes=["BEGIN", "IN", "OUT"]).fit_transform(predictions)]
y_test_bin = ["idiom" if y[0] else "no_idiom" \
              for y in MultiLabelBinarizer().fit_transform(y_test)]

draw_cm(y_test_bin, predictions_bin)


weights_exp = eli5.formatters.explain_weights_dfs(crf, top=30)
weights_exp["targets"][weights_exp["targets"]["target"]=="BEGIN"]
weights_exp["targets"][weights_exp["targets"]["target"]=="IN"]
weights_exp["targets"][weights_exp["targets"]["target"]=="OUT"]


# =============================================================================
# WORD2VEC REGULARIZED
# =============================================================================

print("{}: Creating features for train set...".format(datetime.now()))
X_train = [sent2features(s, word2vec=WORD2VEC) for s in train]
print("{}: Getting labels for train set...".format(datetime.now()))
y_train = [sent2labels(s) for s in train]

print("{}: Creating features for test set".format(datetime.now()))
X_test = [sent2features(s, word2vec=WORD2VEC) for s in test]
print("{}: Getting labels for test set...".format(datetime.now()))
y_test = [sent2labels(s) for s in test]
print("{}: Finished!".format(datetime.now()))

crf = CRF(
    algorithm='lbfgs',
    c1=200,
    c2=0.1,
    max_iterations=20,
    all_possible_transitions=False,
)


crf.fit(X_train, y_train)

predictions = crf.predict(X_test)



report = classification_report(MultiLabelBinarizer().fit_transform(y_test),
                               MultiLabelBinarizer(classes=["BEGIN", "IN", "OUT"]).fit_transform(predictions),
                               target_names=["BEGIN", "IN", "OUT"],
                               digits=3)

print(report)


predictions_bin = ["idiom" if p[0] else "no_idiom" \
                   for p in MultiLabelBinarizer(classes=["BEGIN", "IN", "OUT"]).fit_transform(predictions)]
y_test_bin = ["idiom" if y[0] else "no_idiom" \
              for y in MultiLabelBinarizer().fit_transform(y_test)]

draw_cm(y_test_bin, predictions_bin)



# =============================================================================
# BASIC FEATURES 2 AHEAD REGULARIZED
# =============================================================================


print("{}: Creating features for train set...".format(datetime.now()))
X_train = [sent2features(s, dist=2) for s in train]
print("{}: Getting labels for train set...".format(datetime.now()))
y_train = [sent2labels(s) for s in train]

print("{}: Creating features for test set".format(datetime.now()))
X_test = [sent2features(s, dist=2) for s in test]
print("{}: Getting labels for test set...".format(datetime.now()))
y_test = [sent2labels(s) for s in test]
print("{}: Finished!".format(datetime.now()))
    
    
crf = CRF(
    algorithm='lbfgs',
    c1=200,
    c2=0.1,
    max_iterations=20,
    all_possible_transitions=False,
)

crf.fit(X_train, y_train)


predictions = crf.predict(X_test)

report = classification_report(MultiLabelBinarizer().fit_transform(y_test),
                               MultiLabelBinarizer(classes=["BEGIN", "IN", "OUT"]).fit_transform(predictions),
                               target_names=["BEGIN", "IN", "OUT"],
                               digits=3)

print(report)

predictions_bin = ["idiom" if p[0] else "no_idiom" \
                   for p in MultiLabelBinarizer(classes=["BEGIN", "IN", "OUT"]).fit_transform(predictions)]
y_test_bin = ["idiom" if y[0] else "no_idiom" \
              for y in MultiLabelBinarizer().fit_transform(y_test)]

draw_cm(y_test_bin, predictions_bin)


# =============================================================================
# BASIC FEATURES 3 AHEAD REGULARIZED
# =============================================================================


print("{}: Creating features for train set...".format(datetime.now()))
X_train = [sent2features(s, dist=3) for s in train]
print("{}: Getting labels for train set...".format(datetime.now()))
y_train = [sent2labels(s) for s in train]

print("{}: Creating features for test set".format(datetime.now()))
X_test = [sent2features(s, dist=3) for s in test]
print("{}: Getting labels for test set...".format(datetime.now()))
y_test = [sent2labels(s) for s in test]
print("{}: Finished!".format(datetime.now()))
    
    
crf = CRF(
    algorithm='lbfgs',
    c1=200,
    c2=0.1,
    max_iterations=20,
    all_possible_transitions=False,
)

crf.fit(X_train, y_train)


predictions = crf.predict(X_test)

report = classification_report(MultiLabelBinarizer().fit_transform(y_test),
                               MultiLabelBinarizer(classes=["BEGIN", "IN", "OUT"]).fit_transform(predictions),
                               target_names=["BEGIN", "IN", "OUT"],
                               digits=3)

print(report)

predictions_bin = ["idiom" if p[0] else "no_idiom" \
                   for p in MultiLabelBinarizer(classes=["BEGIN", "IN", "OUT"]).fit_transform(predictions)]
y_test_bin = ["idiom" if y[0] else "no_idiom" \
              for y in MultiLabelBinarizer().fit_transform(y_test)]

draw_cm(y_test_bin, predictions_bin)


# =============================================================================
# PPMI 2 AHEAD REGULARIZED
# =============================================================================


print("{}: Creating features for train set...".format(datetime.now()))
X_train = [sent2features(s, dist=2, include_PPMI=True) for s in train]
print("{}: Getting labels for train set...".format(datetime.now()))
y_train = [sent2labels(s) for s in train]

print("{}: Creating features for test set".format(datetime.now()))
X_test = [sent2features(s, dist=2, include_PPMI=True) for s in test]
print("{}: Getting labels for test set...".format(datetime.now()))
y_test = [sent2labels(s) for s in test]
print("{}: Finished!".format(datetime.now()))
    
    
crf = CRF(
    algorithm='lbfgs',
    c1=200,
    c2=0.1,
    max_iterations=20,
    all_possible_transitions=False,
)

crf.fit(X_train, y_train)


predictions = crf.predict(X_test)

report = classification_report(MultiLabelBinarizer().fit_transform(y_test),
                               MultiLabelBinarizer(classes=["BEGIN", "IN", "OUT"]).fit_transform(predictions),
                               target_names=["BEGIN", "IN", "OUT"],
                               digits=3)

print(report)

predictions_bin = ["idiom" if p[0] else "no_idiom" \
                   for p in MultiLabelBinarizer(classes=["BEGIN", "IN", "OUT"]).fit_transform(predictions)]
y_test_bin = ["idiom" if y[0] else "no_idiom" \
              for y in MultiLabelBinarizer().fit_transform(y_test)]

draw_cm(y_test_bin, predictions_bin)


# =============================================================================
# WORD2VEC 2 AHEAD REGULARIZED
# =============================================================================


print("{}: Creating features for train set...".format(datetime.now()))
X_train = [sent2features(s, dist=2, word2vec=WORD2VEC) for s in train]
print("{}: Getting labels for train set...".format(datetime.now()))
y_train = [sent2labels(s) for s in train]

print("{}: Creating features for test set".format(datetime.now()))
X_test = [sent2features(s, dist=2, word2vec=WORD2VEC) for s in test]
print("{}: Getting labels for test set...".format(datetime.now()))
y_test = [sent2labels(s) for s in test]
print("{}: Finished!".format(datetime.now()))
    
    
crf = CRF(
    algorithm='lbfgs',
    c1=200,
    c2=0.1,
    max_iterations=20,
    all_possible_transitions=False,
)

crf.fit(X_train, y_train)


predictions = crf.predict(X_test)

report = classification_report(MultiLabelBinarizer().fit_transform(y_test),
                               MultiLabelBinarizer(classes=["BEGIN", "IN", "OUT"]).fit_transform(predictions),
                               target_names=["BEGIN", "IN", "OUT"],
                               digits=3)

print(report)

predictions_bin = ["idiom" if p[0] else "no_idiom" \
                   for p in MultiLabelBinarizer(classes=["BEGIN", "IN", "OUT"]).fit_transform(predictions)]
y_test_bin = ["idiom" if y[0] else "no_idiom" \
              for y in MultiLabelBinarizer().fit_transform(y_test)]

draw_cm(y_test_bin, predictions_bin)


# =============================================================================
# PPMI 3 AHEAD REGULARIZED
# =============================================================================


print("{}: Creating features for train set...".format(datetime.now()))
X_train = [sent2features(s, dist=3, include_PPMI=True) for s in train]
print("{}: Getting labels for train set...".format(datetime.now()))
y_train = [sent2labels(s) for s in train]

print("{}: Creating features for test set".format(datetime.now()))
X_test = [sent2features(s, dist=3, include_PPMI=True) for s in test]
print("{}: Getting labels for test set...".format(datetime.now()))
y_test = [sent2labels(s) for s in test]
print("{}: Finished!".format(datetime.now()))
    
    
crf = CRF(
    algorithm='lbfgs',
    c1=200,
    c2=0.1,
    max_iterations=20,
    all_possible_transitions=False,
)

crf.fit(X_train, y_train)


predictions = crf.predict(X_test)

report = classification_report(MultiLabelBinarizer().fit_transform(y_test),
                               MultiLabelBinarizer(classes=["BEGIN", "IN", "OUT"]).fit_transform(predictions),
                               target_names=["BEGIN", "IN", "OUT"],
                               digits=3)

print(report)

predictions_bin = ["idiom" if p[0] else "no_idiom" \
                   for p in MultiLabelBinarizer(classes=["BEGIN", "IN", "OUT"]).fit_transform(predictions)]
y_test_bin = ["idiom" if y[0] else "no_idiom" \
              for y in MultiLabelBinarizer().fit_transform(y_test)]

draw_cm(y_test_bin, predictions_bin)


# =============================================================================
# WORD2VEC 3 AHEAD REGULARIZED
# =============================================================================


print("{}: Creating features for train set...".format(datetime.now()))
X_train = [sent2features(s, dist=3, word2vec=WORD2VEC) for s in train]
print("{}: Getting labels for train set...".format(datetime.now()))
y_train = [sent2labels(s) for s in train]

print("{}: Creating features for test set".format(datetime.now()))
X_test = [sent2features(s, dist=3, word2vec=WORD2VEC) for s in test]
print("{}: Getting labels for test set...".format(datetime.now()))
y_test = [sent2labels(s) for s in test]
print("{}: Finished!".format(datetime.now()))
    
    
crf = CRF(
    algorithm='lbfgs',
    c1=200,
    c2=0.1,
    max_iterations=20,
    all_possible_transitions=False,
)

crf.fit(X_train, y_train)


predictions = crf.predict(X_test)

report = classification_report(MultiLabelBinarizer().fit_transform(y_test),
                               MultiLabelBinarizer(classes=["BEGIN", "IN", "OUT"]).fit_transform(predictions),
                               target_names=["BEGIN", "IN", "OUT"],
                               digits=3)

print(report)

predictions_bin = ["idiom" if p[0] else "no_idiom" \
                   for p in MultiLabelBinarizer(classes=["BEGIN", "IN", "OUT"]).fit_transform(predictions)]
y_test_bin = ["idiom" if y[0] else "no_idiom" \
              for y in MultiLabelBinarizer().fit_transform(y_test)]

draw_cm(y_test_bin, predictions_bin)


# =============================================================================
# WORD2VEC AND PPMI REGULARIZED
# =============================================================================

print("{}: Creating features for train set...".format(datetime.now()))
X_train = [sent2features(s, include_PPMI=True, word2vec=WORD2VEC) for s in train]
print("{}: Getting labels for train set...".format(datetime.now()))
y_train = [sent2labels(s) for s in train]

print("{}: Creating features for test set".format(datetime.now()))
X_test = [sent2features(s, include_PPMI=True, word2vec=WORD2VEC) for s in test]
print("{}: Getting labels for test set...".format(datetime.now()))
y_test = [sent2labels(s) for s in test]
print("{}: Finished!".format(datetime.now()))

crf = CRF(
    algorithm='lbfgs',
    c1=200,
    c2=0.1,
    max_iterations=20,
    all_possible_transitions=False,
)


crf.fit(X_train, y_train)

predictions = crf.predict(X_test)

report = classification_report(MultiLabelBinarizer().fit_transform(y_test),
                               MultiLabelBinarizer(classes=["BEGIN", "IN", "OUT"]).fit_transform(predictions),
                               target_names=["BEGIN", "IN", "OUT"],
                               digits=3)

print(report)


predictions_bin = ["idiom" if p[0] else "no_idiom" \
                   for p in MultiLabelBinarizer(classes=["BEGIN", "IN", "OUT"]).fit_transform(predictions)]
y_test_bin = ["idiom" if y[0] else "no_idiom" \
              for y in MultiLabelBinarizer().fit_transform(y_test)]

draw_cm(y_test_bin, predictions_bin)

# =============================================================================
# WORD2VEC AND PPMI 2 AHEAD REGULARIZED
# =============================================================================

print("{}: Creating features for train set...".format(datetime.now()))
X_train = [sent2features(s, dist=2, include_PPMI=True, word2vec=WORD2VEC) for s in train]
print("{}: Getting labels for train set...".format(datetime.now()))
y_train = [sent2labels(s) for s in train]

print("{}: Creating features for test set".format(datetime.now()))
X_test = [sent2features(s, dist=2, include_PPMI=True, word2vec=WORD2VEC) for s in test]
print("{}: Getting labels for test set...".format(datetime.now()))
y_test = [sent2labels(s) for s in test]
print("{}: Finished!".format(datetime.now()))

crf = CRF(
    algorithm='lbfgs',
    c1=200,
    c2=0.1,
    max_iterations=20,
    all_possible_transitions=False,
)


crf.fit(X_train, y_train)

predictions = crf.predict(X_test)

report = classification_report(MultiLabelBinarizer().fit_transform(y_test),
                               MultiLabelBinarizer(classes=["BEGIN", "IN", "OUT"]).fit_transform(predictions),
                               target_names=["BEGIN", "IN", "OUT"],
                               digits=3)

print(report)


predictions_bin = ["idiom" if p[0] else "no_idiom" \
                   for p in MultiLabelBinarizer(classes=["BEGIN", "IN", "OUT"]).fit_transform(predictions)]
y_test_bin = ["idiom" if y[0] else "no_idiom" \
              for y in MultiLabelBinarizer().fit_transform(y_test)]

draw_cm(y_test_bin, predictions_bin)

# =============================================================================
# WORD2VEC AND PPMI 3 AHEAD REGULARIZED
# =============================================================================

print("{}: Creating features for train set...".format(datetime.now()))
X_train = [sent2features(s, dist=3, include_PPMI=True, word2vec=WORD2VEC) for s in train]
print("{}: Getting labels for train set...".format(datetime.now()))
y_train = [sent2labels(s) for s in train]

print("{}: Creating features for test set".format(datetime.now()))
X_test = [sent2features(s, dist=3, include_PPMI=True, word2vec=WORD2VEC) for s in test]
print("{}: Getting labels for test set...".format(datetime.now()))
y_test = [sent2labels(s) for s in test]
print("{}: Finished!".format(datetime.now()))

crf = CRF(
    algorithm='lbfgs',
    c1=200,
    c2=0.1,
    max_iterations=20,
    all_possible_transitions=False,
)


crf.fit(X_train, y_train)

predictions = crf.predict(X_test)

report = classification_report(MultiLabelBinarizer().fit_transform(y_test),
                               MultiLabelBinarizer(classes=["BEGIN", "IN", "OUT"]).fit_transform(predictions),
                               target_names=["BEGIN", "IN", "OUT"],
                               digits=3)

print(report)


predictions_bin = ["idiom" if p[0] else "no_idiom" \
                   for p in MultiLabelBinarizer(classes=["BEGIN", "IN", "OUT"]).fit_transform(predictions)]
y_test_bin = ["idiom" if y[0] else "no_idiom" \
              for y in MultiLabelBinarizer().fit_transform(y_test)]

draw_cm(y_test_bin, predictions_bin)
















