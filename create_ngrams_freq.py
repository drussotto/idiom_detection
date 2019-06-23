from nltk.corpus import brown, gutenberg, reuters
import pandas as pd
import nltk
import pickle
from utils import get_wordnet_pos
from datetime import datetime
from nltk.collocations import BigramCollocationFinder

OUT_PATH = "./data/{}.pkl"

print("{}: Gathering all the words...".format(datetime.now()))

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

print("{}: Lowercasing all the words...".format(datetime.now()))
words_lower = [w.lower() for w in words]

print("{}: Lematizing all the words...".format(datetime.now()))
wnlt = nltk.WordNetLemmatizer()
words_lemmatized = [wnlt.lemmatize(word, get_wordnet_pos(tb_pos)) \
         for word,tb_pos in nltk.pos_tag(words_lower)]

#bigrams = nltk.collocations.BigramCollocationFinder.from_words(
#        words,
#        window_size=20)

#bigrams.apply_freq_filter(20)
#bigrams_freq = bigrams.ngram_fd


### TODO(?) : Try different windowsizes
print("{}: Creating bigrams frequencies and storing results...".format(datetime.now()))
bigrams_freq = BigramCollocationFinder.from_words(words_lemmatized, window_size=20).ngram_fd

with open(OUT_PATH.format("bigram_freq"), "wb") as f:
    pickle.dump(bigrams_freq, f)


print("{}: Creating unigrams frequencies and storing results...".format(datetime.now()))
unigrams = nltk.FreqDist(words_lemmatized)
unigrams_freq = nltk.FreqDist(words_lemmatized)
#unigrams_freq = {unigram:freq for unigram, freq in unigrams.items() if freq >= 20}

with open(OUT_PATH.format("unigrams_freq"), "wb") as f:
    pickle.dump(unigrams_freq, f)

