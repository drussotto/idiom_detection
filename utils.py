import os.path
from datetime import datetime
import re
from nltk import pos_tag
import random
from math import floor
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


IDIOMS_FILE = "./data/idioms.txt"

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

    ps = pos_tag([word])[0][1]

    return word, ps, tag


def tag_line(line):
    line = line.strip()
    tuples = [word_to_tags(w) for w in line.split(" ")]
    return tuples
            

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

## Taken from Session 5 NER notebook
def word2features(sent, i):
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
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True # Beginning of Sentence

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True #End of sentence

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

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
