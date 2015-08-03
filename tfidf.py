#!/usr/bin/env python
from __future__ import division
import os
import re
import logging
from time import time

from nltk import SnowballStemmer
import pandas as pd
import numpy as np  # Make sure that numpy is imported
import click
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from nltk.corpus import stopwords


def accuracy_among_top(y_true, p_pred):
    """Score function that scores 1 if the target is among top 3 predictions.

    Possible improvements:
     * Score better if the top 1 is the right prediction.
    """
    sorted_classes = sorted(set(y_true))
    return sum([1 for y, probas in zip(y_true, p_pred)
                if sorted_classes.index(y) in np.argsort(probas)[-3:][::-1]]) / len(y_true)


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


def clean_body(questions):
    l = []
    for body in questions:
        l.append(wordlist(body, remove_stopwords=True))
    return l


def bench_(clf, data, params=None):
    """ Given a classifier and data benchmark the performance """

    t0 = time()
    if params is not None:
        clf.set_params(**params)

    print "Fitting with %r" % clf
    clf = clf.fit(data['X_train'], data['y_train'])
    print "Took: %fs" % (time() - t0)

    t1 = time()
    print("Predicting the labels of the testing set")
    prediction = clf.predict(data['X_test'])
    print "Took: %fs" % (time() - t1)

    print "Classification report on test set for classifier:"
    label_to_name = dict(pd.read_csv('./macros.csv.gz', header=0, compression="gzip", usecols=["id", "title"]).values)
    # print("label_to_name", label_to_name)
    labels = sorted(list(set(data['y_test'])))
    print classification_report(data['y_test'], prediction, labels=labels,
                                target_names=[label_to_name[t] for t in labels])

    # cm = confusion_matrix(data['y_test'], prediction)
    # print "Confusion matrix:"
    # print cm


def my_tokenizer(doc):
    return wordlist(doc, remove_stopwords=True, stem=True)


@click.command()
@click.option('-v', '--verbose', count=True)
@click.option('-w', '--train_word2vec', type=click.BOOL, default=False, help="Train word2vec model")
def main(train_word2vec, verbose):
    # Logging for Word2Vec
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Read data from files
    labeled_train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'conversations_labeled.csv.gz'), header=0,
                          compression="gzip")#, nrows=10000)
    print(len(labeled_train))

    # remove the macros that not that popular and insert them into the unlabled training set
    # counts = labeled.groupby('macro_id').count()
    # labeled_train = labeled[labeled.macro_id.apply(lambda r: counts['id'][r] >= 50)]

    # unlabeled_train = labeled[labeled.macro_id.apply(lambda r: counts['id'][r] < 50)]
    # # no need to have the macro_id
    # del unlabeled_train['macro_id']
    #
    # unlabeled_train = pd.concat([
    #     unlabeled_train,
    #     pd.read_csv(os.path.join(os.path.dirname(__file__), 'conversations_unlabeled.csv.gz'), header=0,
    #                 compression="gzip")])

    questions_train, questions_test, macros_train, macros_test = train_test_split(labeled_train['body'],
                                                                                  labeled_train['macro_id'],
                                                                                  test_size=0.4,
                                                                                  random_state=42)

    # # Verify the number of comments that were read
    # print "Read %d labeled train reviews, %d labeled test reviews, " \
    #       "and %d unlabeled reviews\n" % (questions_train.size,
    #                                       questions_test.size,
    #                                       unlabeled_train["body"].size)

    pipe = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('chi2', SelectPercentile(score_func=chi2)),
        ('NB', MultinomialNB()),
    ])

    param_grid = {
        'tfidf__max_df': [0.7],
        'tfidf__tokenizer': [my_tokenizer],
        'tfidf__ngram_range': [(1, 2)],
        'chi2__percentile': [50, 70, 100],
        'NB__alpha': [0.01],
        'NB__fit_prior': [True, None]
    }

    gs_pipe = GridSearchCV(pipe, param_grid, n_jobs=-1)
    # gs_pipe.fit(questions_train, macros_train)

    # params = {
    #     'NB__alpha': 0.01,
    #     'tfidf__ngram_range': (1, 2),
    #     'tfidf__max_df': 0.7,
    #     'tfidf__tokenizer': my_tokenizer,
    #     'NB__fit_prior': True
    # }

    data = {
        'X_train': questions_train,
        'X_test': questions_test,
        'y_train': macros_train,
        'y_test': macros_test
    }

    bench_(gs_pipe, data, params=None)

    print("On train set, accuracy among top 3: ",
          accuracy_among_top(macros_train,
                             gs_pipe.predict_proba(questions_train)))

    print("On test set, accuracy among top 3: ",
          accuracy_among_top(macros_test, gs_pipe.predict_proba(questions_test)))

if __name__ == '__main__':
    main()
