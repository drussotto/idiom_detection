import pandas as pd
from nltk.tokenize import word_tokenize
from utils import write_tagged_sentences, clear_file, num_found_idioms

OUTPUT_FILE = "./data/tagged_sentences_examples.txt"

examples = list(pd.read_csv("./data/idiom_example.csv")["sentence"])



sents = [word_tokenize(sent.replace("’", "'").replace("–", "-")) \
         for sent in examples]

clear_file(OUTPUT_FILE)

success = write_tagged_sentences(sents, OUTPUT_FILE)

if success:
    count = num_found_idioms(OUTPUT_FILE)
    print("number of idioms found (ish): {}".format(count))
