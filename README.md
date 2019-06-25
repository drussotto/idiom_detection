# idiom_detection
Using natural language processing to detect English idioms in documents.

Link to this repo: https://github.com/drussotto/idiom_detection

### Building the datasets

All data files are already loaded into this repo in the data/ folder. To generate the data from
scratch, follow these steps:

1. Run idiom_scraper.py to create the list of idioms (data/idioms.txt)

2. Run idiom_example.py to retrieve the example sentences of idioms (data/idiom_example.csv)

3. Run tag_*.py (brown, gutenberg, reuters, example) to generate sentences where
idioms are tagged (e.g. rank#BEGIN and#IN file#IN)

4. Run build_train_test.py to convert the sentences from the text files into
pickle files containing the lists of sentences. [List-of [List-of (word, POStag, idiomTag)]]

### Preparing to Model

The following steps are necessary to perform, as they create files needed for
modeling that are too large to be stored on github.

1. Get Google's pretrained Word2Vec file here https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit. Unzip the .gz file and save it in the data/
folder.

2. Run create_n_grams_freq.py to create pickle files that contain the frequencies
of unigrams and bigrams in the train and test data.

### The Modeling Process

* "Model Testing.ipynb" contains a "grid search" of different possible combinations
of using Word2Vec similarity scores, PMI/PPMI similarity scores, and how many
words ahead and behind to look at.

* hyperparameter_tuning.py does a formal randomized grid search to see what the
"best" regularization terms are.

* regularization.py performs more intense regularization in an attempt to drive
overly specific features down to zero (the result is never predicting the presence
  of an idiom)
