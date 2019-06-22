from nltk.corpus import brown, gutenberg, reuters
import pandas as pd
import nltk
import pickle
from utils import get_wordnet_pos

OUT_PATH = "./data/{}.pkl"

idiom_examples = pd.read_csv("./data/idiom_example.csv")["sentence"]

idiom_examples_split = [nltk.word_tokenize(sent) for sent in idiom_examples]

examples_words = [word for sent in idiom_examples_split \
                  for word in sent]

# this doesn't work, a little salty about it
#examples_words = [word for word_tokenize(sent) in idiom_examples \
#                  for word in word_tokenize(sent)]

# Does it matter that the last sentence of one document will be combined with 
# the first sentence of another?
words = brown.words() + gutenberg.words() + reuters.words() + examples_words

words = [w.lower() for w in words]

wnlt = nltk.WordNetLemmatizer()
words = [wnlt.lemmatize(word, get_wordnet_pos(tb_pos)) \
         for word,tb_pos in nltk.pos_tag(words)]

#bigrams = nltk.collocations.BigramCollocationFinder.from_words(
#        words,
#        window_size=20)

## How to choose window size?? and what is a noncontiguous bigram??

#bigrams.apply_freq_filter(20)
#bigrams_freq = bigrams.ngram_fd


bigrams_freq = nltk.collocations.BigramCollocationFinder.from_words(words).ngram_fd



with open(OUT_PATH.format("bigram_freq"), "wb") as f:
    pickle.dump(bigrams_freq, f)



unigrams = nltk.FreqDist(words)
unigrams_freq = nltk.FreqDist(words)
#unigrams_freq = {unigram:freq for unigram, freq in unigrams.items() if freq >= 20}

with open(OUT_PATH.format("unigrams_freq"), "wb") as f:
    pickle.dump(unigrams_freq, f)

