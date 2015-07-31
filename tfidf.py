#!/usr/bin/env python

import os
import re
import logging
from time import time

from nltk import SnowballStemmer
import pandas as pd
import numpy as np  # Make sure that numpy is imported
import click
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from nltk.corpus import stopwords


def wordlist(body, remove_stopwords=False, stem=False):
    """ convert a document to a sequence of words, optionally removing stop words.  Returns a list of words."""

    # Remove non-letters
    text = re.sub("[^a-zA-Z]", " ", body)

    # convert words to lower case and split them
    words = text.lower().split()

    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    if stem:
        stemmer = SnowballStemmer("english")
        words = [stemmer.stem(w) for w in words]

    return words


def to_sentences(review, tokenizer, remove_stopwords=False):
    """Function to split a review into parsed sentences.
    Returns a list of sentences, where each sentence is a list of words

    """

    # Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.decode('utf8').strip())

    # Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append(wordlist(raw_sentence, remove_stopwords))
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences


def feature_vector(words, model, num_features):
    """ Function to average all of the word vectors in a given paragraph

    """

    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,), dtype="float32")

    nwords = 0

    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.index2word)


    # Loop over each word in the and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec, model[word])

    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec, nwords)

    return featureVec


def average_vec(questions, model, num_features):
    # Given a set of question (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array

    counter = 0

    # Preallocate a 2D numpy array, for speed
    feature_vectors = np.zeros((len(questions), num_features), dtype="float32")

    # Loop through the reviews
    for question in questions:
        if counter % 1000 == 0:
            print "Question %d of %d" % (counter, len(questions))

        # Call the function (defined above) that makes average feature vectors
        fv = feature_vector(question, model, num_features)
        if np.isfinite(fv).all():
            feature_vectors[counter] = fv

        counter += 1

    return feature_vectors


def clean_body(questions):
    l = []
    for body in questions:
        l.append(wordlist(body, remove_stopwords=True))
    return l


def bench_(clf, params, data):
    """ Given a classifier and data benchmark the performance """

    t0 = time()
    clf.set_params(**params)
    print "Fitting with %r" % clf
    clf = clf.fit(data['X_train'], data['y_train'])
    print "Took: %fs" % (time() - t0)

    t1 = time()
    print("Predicting the of the testing set")
    prediction = clf.predict(data['X_test'])
    print "Took: %fs" % (time() - t1)

    print "Classification report on test set for classifier:"
    print classification_report(data['y_test'], prediction)

    # cm = confusion_matrix(data['y_test'], prediction)
    # print "Confusion matrix:"
    # print cm


@click.command()
@click.option('-v', '--verbose', count=True)
@click.option('-w', '--train_word2vec', type=click.BOOL, default=False, help="Train word2vec model")
def main(train_word2vec, verbose):
    # Logging for Word2Vec
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Read data from files
    labeled = pd.read_csv(os.path.join(os.path.dirname(__file__), 'conversations_labeled.csv.gz'), header=0,
                          compression="gzip")

    # remove the macros that not that popular and insert them into the unlabled training set
    counts = labeled.groupby('macro_id').count()
    labeled_train = labeled[labeled.macro_id.apply(lambda r: counts['id'][r] >= 50)]
    unlabeled_train = labeled[labeled.macro_id.apply(lambda r: counts['id'][r] < 50)]
    # no need to have the macro_id
    del unlabeled_train['macro_id']

    unlabeled_train = pd.concat([
        unlabeled_train,
        pd.read_csv(os.path.join(os.path.dirname(__file__), 'conversations_unlabeled.csv.gz'), header=0,
                    compression="gzip")])

    questions_train, questions_test, macros_train, macros_test = train_test_split(labeled_train['body'],
                                                                                  labeled_train['macro_id'],
                                                                                  test_size=0.4,
                                                                                  random_state=42)

    # Verify the number of comments that were read
    print "Read %d labeled train reviews, %d labeled test reviews, " \
          "and %d unlabeled reviews\n" % (questions_train.size,
                                          questions_test.size,
                                          unlabeled_train["body"].size)

    def my_tokenizer(doc):
        return wordlist(doc, remove_stopwords=True, stem=True)

    pipe = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('NB', MultinomialNB()),
    ])

    param_grid = {
        'NB__alpha': [0.01, 0.1],
        'tfidf__max_df': [0.7],
        'tfidf__tokenizer': [my_tokenizer],
        'tfidf__ngram_range': [(1, 2), (1, 3)]
    }

    #gs_pipe = GridSearchCV(pipe, param_grid)
    #gs_pipe.fit(questions_train, macros_train)

    params = {
        'NB__alpha': 0.01,
        'tfidf__ngram_range': (1, 2),
        'tfidf__max_df': 0.7,
        'tfidf__tokenizer': my_tokenizer
    }

    data = {
        'X_train': questions_train,
        'X_test': questions_test,
        'y_train': macros_train,
        'y_test': macros_test
    }

    bench_(pipe, params, data)


if __name__ == '__main__':
    main()
