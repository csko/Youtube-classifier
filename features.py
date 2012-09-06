#!/usr/bin/env python2
# -*- coding: utf-8 -*

import nltk
import itertools
import numpy as np
import re
import sys
import codecs

from parse import load_data, PROJECT

from nltk.collocations import TrigramCollocationFinder
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.metrics import TrigramAssocMeasures
from nltk.tokenize import sent_tokenize
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer

from sklearn import cross_validation
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


http_re = re.compile(r"((http|ftp|https):\/\/)?[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?", re.IGNORECASE)
oddities_re = re.compile(ur"(=|¡¿|·|\\|\^|~|…|“|”|ß|€)")
tokenize2_re = re.compile(r"(\w+)([-\*.\\/#])+", re.UNICODE)
# TODO: re: phone number

bad_words = set([line.strip().lower() for line in open(PROJECT + 'db/badwords.txt')])

def parse_text(text):
    # TODO: regex split
    text = text.replace("_", " ").replace("\r\n", " NL2 ").replace("\n", " NL ") \
            .replace("\'", "'").replace(u"\xc2\xa0", " NBSP ") \
            .replace(u"\xa0", " NBSP2 ")

    text = text.replace(u"\xe2\x80\x98", "'")
    text = text.replace(u"\xe2\x80\x99", "'")
    text = text.replace(u"\xe2\x80\x9c", "\"")
    text = text.replace(u"\xe2\x80\x9d", "\"")
    text = text.replace(u"\xe2\x80\x93", "-")
    text = text.replace(u"\xe2\x80\x94", "--")
    text = text.replace(u"\xe2\x80\xa6", "...")

    text = text.replace(u"’", "'")
    text = text.replace(u"`", "'")

    text = http_re.sub(" dummyhtml ", text)
#    text = dotdot_re.sub(lambda m: "%s %s %s" % (m.group(1), m.group(2), m.group(4)), text)

#    text = youre_re.sub("you're", text) # TODO: proper split
    text = oddities_re.sub(lambda m: " %s " % m.group(1), text)
    text = tokenize2_re.sub(lambda m: "%s %s " % (m.group(1), m.group(2)), text)
    return text

def create_features(X, user_data):
    res = []

    wordtokenizer = TreebankWordTokenizer()
    wnl = WordNetLemmatizer()
    ps = nltk.stem.PorterStemmer()
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())

    for date, comment, user in X:
        feat = {}

#        comment = comment.lower()

        comment = parse_text(comment)

        sents = sent_tokenize(comment)
        doc = []
        for sent in sents:
            # Tokenize each sentence.
            doc += wordtokenizer.tokenize(sent)

        for i, word in enumerate(doc):
#            if doc[i] in bad_words:
#                doc[i] = '_badword_'
            doc[i] = ps.stem(doc[i])
            doc[i] = wnl.lemmatize(doc[i])
#            if doc[i] in bad_words:
#                doc[i] = '_badword_'

#        trigram_finder = TrigramCollocationFinder.from_words(comment)
#        trigrams = trigram_finder.nbest(TrigramAssocMeasures.chi_sq, n=10)
        bigram_finder = BigramCollocationFinder.from_words(doc)
        bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, n=12)

#        trigram = dict([(ngram, True) for ngram in itertools.chain(comment, trigrams)])
        bigram = dict([(ngram, True) for ngram in itertools.chain(doc, bigrams)])

#        feat.update(trigram)
        feat.update(bigram)


        text_vocab = set(w for w in doc if w.isalpha())
        unusual = text_vocab.difference(english_vocab)
        unusual_ratio = len(unusual) / len(text_vocab) if len(text_vocab) != 0 else -1.0

        user_info = user_data[user]

        has_bad_word = True
        for word in bad_words:
            if word in comment.lower():
                break
        else:
            has_bad_word = False

        def n_none(x):
            return int(x) if x is not None else 0
        def c_none(x):
            return x if x is not None else "__None__"

        feat['_AlwaysPresent'] = True
        feat['_word_num'] = len(doc)
#        feat['_sent_num'] = len(sents)
        feat['_word_var'] = len(set(doc)) / len(doc)
#        feat['_sent_var'] = len(set(sents)) / len(sents)
        feat['_unusual_ratio'] = unusual_ratio
#        feat['_username'] = user
        feat['_user_subcount'] = user_info['SubscriberCount']
        feat['_user_friends'] = user_info['FriendsAdded']
        feat['_user_favs'] = user_info['VideosFavourited']
        feat['_user_videorates'] = user_info['VideosRated']
        feat['_user_videouploads'] = user_info['VideosUploaded']
        feat['_user_videocomments'] = user_info['VideosCommented']
        feat['_user_videoshares'] = user_info['VideosShared']
        feat['_user_usersubs'] = user_info['UserSubscriptionsAdded']
        feat['_user_gender'] =  c_none(user_info['Gender'])
        feat['_user_age'] =  n_none(user_info['Age'])
        feat['_user_closed'] = user_info['UserAccountClosed']
        feat['_user_suspended'] = user_info['UserAccountSuspended']
        feat['_user_has_school'] = 1 if user_info['School'] is not None else 0
        feat['_user_has_books'] = 1 if user_info['Books'] is not None else 0
        feat['_user_has_movies'] = 1 if user_info['Movies'] is not None else 0
        feat['_user_has_music'] = 1 if user_info['Music'] is not None else 0
        feat['_user_has_location'] = 1 if user_info['Location'] is not None else 0
        feat['_user_has_hometown'] = 1 if user_info['Hometown'] is not None else 0
#        feat['_user_last'] = user_info['LastWebAccess']
        feat['_has_bad_word'] = has_bad_word

#        print feat
        res.append(feat)
    return res

def select_rows(iterable, rows):
    res = []
    for j, row in enumerate(iterable):
        if j in rows:
            res.append(row)
    return res

def kfold_run((i, k, cls, train_X, train_y, test_X, test_y)):
    print "Training on fold #%d/%d" % (i + 1, k)
    cls.fit(train_X, train_y)
    return cls.score(test_X, test_y)

def main():
    sys.stdout = codecs.getwriter('utf8')(sys.stdout)

    print "Loading data."
    videos, users, reviews = load_data()

    print "Extracting features."
    feats = create_features([(x['date'], x['text'], x['user']) for x in reviews], users)
    #y = np.array([1 if x['spam'] == 'true' else 0 for x in reviews])
    y = np.array([1 if x['adult'] == 'true' else 0 for x in reviews])

    print "Vectorizing features."
    v = DictVectorizer(sparse=False)
    feats = v.fit_transform(feats)

    print "Starting K-fold cross validation."
    k = 10
    cv = cross_validation.KFold(len(feats), k=k, indices=True)

    from sklearn.ensemble import RandomForestClassifier

    cls = LogisticRegression(penalty='l2', tol=0.00001, fit_intercept=False, dual=False, C=2.4105, class_weight=None)

    scores = cross_validation.cross_val_score(cls, feats, y, cv=10, score_func=metrics.f1_score)
    for i, score in enumerate(scores):
        print "Fold %d: %.5f" % (i, score)
    print "Mean score: %0.5f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)

if __name__ == "__main__":
    main()
