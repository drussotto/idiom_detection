import os.path
from datetime import datetime
import re
from nltk import pos_tag, WordNetLemmatizer
import random
from math import floor, log
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pickle
from nltk.corpus import wordnet
import os
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import eli5

IDIOMS_FILE = "./data/idioms.txt"

if os.path.exists("./data/unigrams_freq.pkl"):
    with open("./data/unigrams_freq.pkl", "rb") as f:
        UNIGRAM_FREQ = pickle.load(f)
        UNIGRAM_FREQ_COUNT = sum(UNIGRAM_FREQ.values())


if os.path.exists("./data/bigram_freq.pkl"):
    with open("./data/bigram_freq.pkl", "rb") as f:
        BIGRAM_FREQ = pickle.load(f)
        BIGRAM_FREQ_COUNT = sum(BIGRAM_FREQ.values())
        
## Empirically this is expensive to create for some reason
WNLT = WordNetLemmatizer()

def text_has_idiom(text, idiom):
    text = text.lower()
    idiom_start = text.find(idiom)
    
    if idiom_start == -1:
        return False
    else:

        try:
            alpha_before = idiom_start != 0 and re.match("[a-zA-Z]", text[idiom_start-1])
        except IndexError:
            alpha_before = False
        try:
            alpha_after = re.match("[a-zA-Z]", text[idiom_start + len(idiom)])
        except IndexError:
            alpha_after = False
        
        if alpha_before or alpha_after:
            return False
    
        return True
        
        
def tag_idiom(idiom):
    idiom_as_list = idiom.split(" ")
    
    for i,word in enumerate(idiom_as_list):
        if i == 0:
            idiom_as_list[i] = idiom_as_list[i] + "#BEGIN"
        else:
            idiom_as_list[i] = idiom_as_list[i] + "#IN"
            
    tagged = " ".join(idiom_as_list)
    
    return tagged

def clear_file(output_file):
    if os.path.exists(output_file):
        with open(output_file, "w") as f:
            f.write("")

def retain_case(word, word_lower):    
    # This can only happen from a previous tag
    if "#" in word:
        return word
    
    if "#" in word_lower:
        _,tag = word_lower.split("#")
        word = word + "#" + tag
    
    return word

def replace_with_tag(text, idiom):
    text_lower = text.lower()
    
    if text_has_idiom(text_lower, idiom):
        text_lower = text_lower.replace(idiom, tag_idiom(idiom))

        text_as_list = [retain_case(word, word_lower) \
                        for word, word_lower in zip(text.split(),
                                                    text_lower.split())
                        ]
    
        text = " ".join(text_as_list)
    
    return text

#KNOWN BUG(?): Information loss - everything is lowercased

#KNOWN BUG: idiom in one context, not another.
# e.g.: "all the same, ..." (an idiom) vs "they are not all the same" (not an idiom) 
            
def write_tagged_sentences(sents, output_file, idioms_file=IDIOMS_FILE):
    print("Starting script execution: {}".format(datetime.now()))          
    
    for i in range(len(sents)):
        # enumerate breaks the NLTK code for some reason, so had to do it this way
        sent = sents[i]
        
        if i % 500 == 0:
            print("{}: Tagged {} of {} sentences..."
                  .format(datetime.now(),i, len(sents)))
        
        sent_text = " ".join(sent)
        
        with open(idioms_file, "r") as f:
            for line in f:
                idiom = line.lower().strip()
                sent_text = replace_with_tag(sent_text, idiom)
                
        with open(output_file, "a") as f:
            f.write(sent_text + "\n")
                    
    print("Finishing script execution: {}".format(datetime.now()))
    
    return True
    
    
def num_found_idioms(file):
    with open(file, "r") as f:
        count = 0
        for line in f:
            if "#BEGIN" in line:
                count += 1
    
    return count


def word_to_tags(word):
    if "#" in word:
        word, tag = word.split("#")
    else:
        tag = "OUT"

    return word, tag


def add_postags(words_w_tags):
    words = [word for word, itag in words_w_tags]
    itags = [itag for word, itag in words_w_tags]
    postags = [postag for word, postag in pos_tag(words)]
    
    return list(zip(words, postags, itags))
    

def tag_line(line):
    return add_postags([word_to_tags(w) for w in line.strip().split()])
    

            

def sent_has_idiom(sent):
    for _, _, itag in sent:
        if itag == "BEGIN":
            return True
    
    return False

def stratified_train_test(tagged_sentences,
                          seed=floor(datetime.now().timestamp()),
                          train_pct=0.8,
                          undersample_factor=4):
    with_idioms = [sent for sent in tagged_sentences if sent_has_idiom(sent)]
    wo_idioms = [sent for sent in tagged_sentences if not sent_has_idiom(sent)]
    
    random.seed(seed)
    
    train_target = random.sample(with_idioms,
                                 floor(len(with_idioms)*train_pct))
    test_target = random.sample(with_idioms,
                                floor(len(with_idioms)*(1-train_pct)))
    

    undersampled = random.sample(wo_idioms,len(with_idioms)*undersample_factor)    
    train_without = random.sample(undersampled,
                                  floor(len(undersampled)*train_pct))    
    test_without = random.sample(undersampled,
                                 floor(len(undersampled)*(1-train_pct)))
    
    train = train_target + train_without
    test = test_target + test_without
    
    random.shuffle(train)
    random.shuffle(test)
    
    return train, test



def add_prev_word_features(sent, word_index, features,
                           dist=1, include_PMI=False,
                           include_PPMI=False, word2vec=None):
    if word_index == 0:
        features['BOS'] = True # Beginning of Sentence
        return features
        
    for i in range(1, dist+1):
        
        if word_index - i < 0:
            return features
        
        prev_word, prev_postag, _ = sent[word_index-i]

        features.update({
            '-{}:word.lower()'.format(i): prev_word.lower(),
            '-{}:word.istitle()'.format(i): prev_word.istitle(),
            '-{}:word.isupper()'.format(i): prev_word.isupper(),
            '-{}:postag'.format(i): prev_postag,
            '-{}:postag[:2]'.format(i): prev_postag[:2],
        })
    
        if include_PMI:
            pmi_i = calc_pmi((sent[word_index][0],sent[word_index][1]),
                             (prev_word, prev_postag))
            
            features["-{}:pmi".format(i)] = pmi_i
        
        if include_PPMI:
            ppmi_i = calc_ppmi((sent[word_index][0],sent[word_index][1]),
                             (prev_word, prev_postag))
            
            features["-{}:ppmi".format(i)] = ppmi_i
        
        if word2vec:
            w2v_i = calc_word2vec_similarity(
                    (sent[word_index][0],sent[word_index][1]),
                    (prev_word, prev_postag),
                    word2vec)
            
            features["-{}:word2vec".format(i)] = w2v_i

    return features

def add_next_word_features(sent, word_index, features,
                           dist=1, include_PMI=False,
                           include_PPMI=False, word2vec=None):
    if word_index == len(sent)-1:
        features['EOS'] = True # End of Sentence
        return features
        
    for i in range(word_index+1, word_index+dist+1):
        
        if i == len(sent):
            return features
        
        next_word, next_postag, _ = sent[i]

        features.update({
            '+{}:word.lower()'.format(i-word_index): next_word.lower(),
            '+{}:word.istitle()'.format(i-word_index): next_word.istitle(),
            '+{}:word.isupper()'.format(i-word_index): next_word.isupper(),
            '+{}:postag'.format(i-word_index): next_postag,
            '+{}:postag[:2]'.format(i-word_index): next_postag[:2],
        })
    
        if include_PMI:
            pmi_i = calc_pmi((sent[word_index][0], sent[word_index][1]),
                             (next_word, next_postag))
            features["+{}:pmi".format(i-word_index)] = pmi_i
        
        if include_PPMI:
            ppmi_i = calc_ppmi((sent[word_index][0], sent[word_index][1]),
                             (next_word, next_postag))
            features["+{}:ppmi".format(i-word_index)] = ppmi_i
            
        if word2vec:
            w2v_i = calc_word2vec_similarity(
                    (sent[word_index][0],sent[word_index][1]),
                    (next_word, next_postag),
                    word2vec)
            
            features["+{}:word2vec".format(i)] = w2v_i

    return features


## Adapted from Session 5 NER notebook
def word2features(sent, i, dist=1, include_PMI=False,
                  include_PPMI=False, word2vec=None):
    
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],         #SUFFIX (ing... what else?)
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    

    features = add_prev_word_features(sent, i, features,
                                      dist=dist,include_PMI=include_PMI,
                                      include_PPMI=include_PPMI,
                                      word2vec=word2vec)
    
    features = add_next_word_features(sent, i, features,
                                      dist=dist,include_PMI=include_PMI,
                                      include_PPMI=include_PPMI,
                                      word2vec=word2vec)

    return features




def sent2features(sent, dist=1, include_PMI=False,
                  include_PPMI=False, word2vec=None):
    
    return [word2features(sent, i, dist, include_PMI,
                          include_PPMI, word2vec) \
            for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def draw_cm(actual_label, prediction_label):   
    # create a confusion matrix for the true target value and the predicted target value
    cm = confusion_matrix(y_true=actual_label, y_pred=prediction_label)
    # config the plot
    plt.figure(figsize=(8,8)) 
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues) # plt.cm.Blues
    plt.tight_layout()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["idiom", "no_idiom"])
    plt.yticks(tick_marks, ["idiom", "no_idiom"])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # add a layer to the plot describes the count for each pair of true value and prediction
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(x=j, y=i, s=int(cm[i, j]), va='center', ha='center', 
                     color='black', fontsize=20)


def lower_lemmatize(word, tb_pos):
    lower = word.lower()
    
    pos = get_wordnet_pos(tb_pos)
    
    return WNLT.lemmatize(lower, pos)


#adapted from class notebook
def calc_pmi(word_w_tag1, word_w_tag2):
    word1, pos1 = word_w_tag1
    word2, pos2 = word_w_tag2
    
    word1 = lower_lemmatize(word1, pos1)
    word2 = lower_lemmatize(word2, pos2)
    
    marginal_word1 = float(UNIGRAM_FREQ[word1]) / UNIGRAM_FREQ_COUNT
    marginal_word2 = float(UNIGRAM_FREQ[word2]) / UNIGRAM_FREQ_COUNT
 
    joint_w1_w2 = float(BIGRAM_FREQ[(word1, word2)])/ BIGRAM_FREQ_COUNT
    
    # This technically shouldn't happen - need to do postagging in a different way
    if marginal_word1 == 0 or marginal_word2 == 0:
        return 0
    
    pmi = round(log(max(0.0005, joint_w1_w2/(marginal_word1*marginal_word2)),2),2)
    
    return pmi

def calc_ppmi(word_w_tag1, word_w_tag2):
    pmi = calc_pmi(word_w_tag1, word_w_tag2)
    return max(0, pmi)


## Does it make sense to lemmatize here? Does it for PMI? 
## I think so, we undersampled like crazy so our dataset isnt THAT big...
## https://stackoverflow.com/questions/23877375/word2vec-lemmatization-of-corpus-before-training
#def calc_word2vec_similarity(word1, word2):
#    return WORD2VEC.similarity(word1, word2)

def calc_word2vec_similarity(word_w_tag1, word_w_tag2, model):
    word1, pos1 = word_w_tag1
    word2, pos2 = word_w_tag2
    
    word1 = lower_lemmatize(word1, pos1)
    word2 = lower_lemmatize(word2, pos2)
    
    # doesn't contain stopwords, punctuation, numbers, and other words
    try:
        retval = model.similarity(word1, word2)
    except KeyError:
        retval = 0
    
    return retval


# Note: taken from class notebook
def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN #by default is noun

def create_features(train, test, dist=1, include_PMI=False,
                    include_PPMI=False, word2vec=None):
    
    print("{}: Creating features for train set...".format(datetime.now()))
    X_train = [sent2features(s,dist=dist, include_PMI=include_PMI,
                             include_PPMI=include_PPMI, word2vec=word2vec)\
                for s in train]
    
    print("{}: Getting labels for train set...".format(datetime.now()))
    y_train = [sent2labels(s) for s in train]
    
    print("{}: Creating features for test set".format(datetime.now()))
    X_test = [sent2features(s,dist=dist, include_PMI=include_PMI,
                             include_PPMI=include_PPMI, word2vec=word2vec)\
                for s in test]
    print("{}: Getting labels for test set...".format(datetime.now()))
    y_test = [sent2labels(s) for s in test]
    
    print("{}: Finished!".format(datetime.now()))
    
    return X_train, y_train, X_test, y_test


def binary_f1(obs, pred):
    pred_bin = [p[0] for p in MultiLabelBinarizer(classes=["BEGIN", "IN", "OUT"])\
                               .fit_transform(pred)]

    obs_bin = [y[0] for y in MultiLabelBinarizer(classes=["BEGIN", "IN", "OUT"])\
                               .fit_transform(obs)]
    
    return f1_score(obs_bin, pred_bin)


def print_classification_report(pred, obs):
    report = classification_report(
            MultiLabelBinarizer().fit_transform(obs),
            MultiLabelBinarizer(classes=["BEGIN", "IN", "OUT"]).fit_transform(pred),
            target_names=["BEGIN", "IN", "OUT"],
            digits=3)
    
    print(report)
    
    return report

def binarized_confusion_matrix(pred, obs):
    pred_bin = ["idiom" if p[0] else "no_idiom" \
                   for p in MultiLabelBinarizer(classes=["BEGIN", "IN", "OUT"])\
                               .fit_transform(pred)]

    obs_bin = ["idiom" if y[0] else "no_idiom" \
                  for y in MultiLabelBinarizer().fit_transform(obs)]
    
    draw_cm(obs_bin, pred_bin)
    
def explain_weights(crf, html=False, top=30):
    if html:
        return eli5.show_weights(crf, top=top)
    else:
        return eli5.formatters.explain_weights_dfs(crf, top=top)
#        weights_exp = eli5.formatters.explain_weights_dfs(crf, top=30)
#        weights_exp["targets"][weights_exp["targets"]["target"]=="BEGIN"]
#        weights_exp["targets"][weights_exp["targets"]["target"]=="IN"]
#        weights_exp["targets"][weights_exp["targets"]["target"]=="OUT"]



#mtx = multilabel_confusion_matrix(MultiLabelBinarizer().fit_transform(y_test),
#                                  MultiLabelBinarizer().fit_transform(predictions))
#
#
#print(mtx)



