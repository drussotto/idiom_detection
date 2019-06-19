from datetime import datetime
import pickle
from utils import tag_line, stratified_train_test

INPUT = ["./data/tagged_sentences_brown.txt",
         "./data/tagged_sentences_reuters.txt",
         "./data/tagged_sentences_gutenberg.txt",
         "./data/tagged_sentences_examples.txt"]

TRAIN_PCT = 0.8
UNDERSAMPLE_FACTOR = 4
OUT_PATH = "./data/{}.pkl"
SEED = 20190619


print("{}: Reading tagged sentences...".format(datetime.now()))
tagged_sentences = []
for fname in INPUT:
    with open(fname, "r") as f: 
        print("{}: Tagging sentences from {}...".format(datetime.now(), fname))
        ts = [tag_line(line) for line in f]
        print("{}: Finished!".format(datetime.now()))
       
        print("{}: Adding newly tagged sentences to rest of sentences..."
                      .format(datetime.now()))
        tagged_sentences.extend(ts)
        print("{}: Finished!".format(datetime.now()))
        

print("{}: Creating train and test datasets...".format(datetime.now()))

train, test = stratified_train_test(tagged_sentences,
                                    SEED,
                                    TRAIN_PCT,
                                    UNDERSAMPLE_FACTOR)

print("{}: Writing train data to file via pickle ...".format(datetime.now()))

with open(OUT_PATH.format("train"), "wb") as f:
    pickle.dump(train, f)

print("{}: Writing test data to file via pickle ...".format(datetime.now()))

with open(OUT_PATH.format("test"), "wb") as f:
    pickle.dump(test, f)

print("{}: Finished!".format(datetime.now()))
    