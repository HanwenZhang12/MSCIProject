### Contains Classic Feature-based Models and CNN

import json

from keras.src.layers import MaxPooling1D
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from keras.layers import Dense
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from torch.utils.data import DataLoader, Dataset



def read_train_file():
    file_path = './data/train.txt'
    train = []
    with open(file_path, "r",  encoding="utf-8") as file:
        for line in file:
            data = json.loads(line.lower())
            train.append(data)
    print('Train data has been read successfully.')
    return train

def read_test_file():
    file_path = './data/val.txt'
    test = []
    with open(file_path, "r",  encoding="utf-8") as file:
        for line in file:
            data = json.loads(line.lower())
            test.append(data)
    print('Test data has been read successfully.')
    return test

def preprocess_sentence(sentence):
    tokens = word_tokenize(sentence)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [token for token in tokens if token.isalpha()]
    res = ''
    for token in tokens:
        res = res + token + ' '
    return res[:-1]

def init_model(train, test, data):
    train_x = []
    train_y = []
    test_x = []
    test_y = []

    if data == 'title':
        for t in train:
            # train_x.append(preprocess_sentence(t['posttext'][0]))
            sentence = preprocess_sentence(t['posttext'][0])
            # sentence = sentence + '\n' + ' '
            # for p in t['targetparagraphs']:
            #     sentence = sentence + preprocess_sentence(p) + ' '
            sentence = sentence[:-1]
            train_x.append(sentence)
            train_y.append(t['tags'][0])

        for t in test:
            # test_x.append(preprocess_sentence(t['posttext'][0]))
            sentence = preprocess_sentence(t['posttext'][0])
            # sentence = sentence + '\n' + ' '
            # for p in t['targetparagraphs']:
            #     sentence = sentence + preprocess_sentence(p) + ' '
            sentence = sentence[:-1]
            test_x.append(sentence)
            test_y.append(t['tags'][0])
    else:
        for t in train:
            sentence = ''
            for paragraph in t['targetparagraphs']:
                sentence = sentence + paragraph + ' '
            sentence = sentence[:-1]
            train_x.append(preprocess_sentence(sentence))
            train_y.append(t['tags'][0])

        for t in test:
            sentence = ''
            for paragraph in t['targetparagraphs']:
                sentence = sentence + paragraph + ' '
            sentence = sentence[:-1]
            test_x.append(preprocess_sentence(sentence))
            test_y.append(t['tags'][0])
    ### Word2vec
    # vectorizer = CountVectorizer(ngram_range=(2, 2))
    # corpus_features = vectorizer.fit_transform(train_x)
    # corpus_test_features = vectorizer.transform(test_x)
    # classifier = MultinomialNB()
    # classifier.fit(corpus_features, train_y)
    # pred = classifier.predict(corpus_test_features)
    # corpus_test_accuracy = accuracy_score(test_y, pred)
    # print(str(corpus_test_accuracy))

    # ### Naive Bayes
    # model2 = make_pipeline(TfidfVectorizer(), MultinomialNB())
    # model2.fit(train_x, train_y)
    # spoiler_types_pred = model2.predict(test_x)
    # print(accuracy_score(test_y, spoiler_types_pred))

    ### CNN
    # le = LabelEncoder()
    # spoiler_types = le.fit_transform(train_y)
    # spoiler_types = to_categorical(spoiler_types)
    # tokenizer = Tokenizer(num_words=5000)
    # tokenizer.fit_on_texts(train_x)
    # sequences = tokenizer.texts_to_sequences(train_x)
    # data = pad_sequences(sequences, maxlen=500)
    # X_train, X_test, y_train, y_test = train_test_split(data, spoiler_types, test_size=0.2, random_state=42)
    # model3 = Sequential()
    # model3.add(Embedding(5000, 200, input_length=500))
    # model3.add(Conv1D(128, 4, activation='relu'))
    # model3.add(GlobalMaxPooling1D())
    # model3.add(Dense(3, activation='softmax'))
    # model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model3.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    # pred = []
    # for i in range(0, len(test_x)):
    #     new_post = test_x[i]
    #     new_post_seq = tokenizer.texts_to_sequences([new_post])
    #     new_post_data = pad_sequences(new_post_seq, maxlen=500)
    #     pred3 = model3.predict(new_post_data)
    #     pred3 = le.inverse_transform([pred3.argmax()])[0]
    #     pred.append(pred3)
    # print(accuracy_score(test_y, pred))

    ## Binary Naive Bayes
    # _train_y = []
    # for y in train_y:
    #     if y == 'multi':
    #         _train_y.append('multi')
    #     else:
    #         _train_y.append('n')
    # model1 = make_pipeline(TfidfVectorizer(), MultinomialNB())
    # model1.fit(train_x, _train_y)
    #
    # model2 = make_pipeline(TfidfVectorizer(), MultinomialNB())
    # i = 0
    # while i < len(train_x):
    #     if train_y[i] == 'multi':
    #         del train_x[i]
    #         del train_y[i]
    #     else:
    #         i+=1
    # model2.fit(train_x, train_y)
    # pred = []
    # for x in test_x:
    #     p = model1.predict([x])
    #     if p[0] == 'multi':
    #         pred.append(p[0])
    #     else:
    #         p = model2.predict([x])
    #         pred.append(p[0])
    # print(accuracy_score(test_y, pred))

    # ### logic regression
    # sentences = train_x
    # y = train_y
    # vectorizer = CountVectorizer()
    # X = vectorizer.fit_transform(sentences)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # clf = LogisticRegression(max_iter=5000)
    # clf.fit(X_train, y_train)
    # print('Accuracy:', clf.score(X_test, y_test))

def build_model(train, test):
    title = init_model(train, test, 'title')
    #paragraph = init_model(train, test, 'paragraph')

if __name__ == '__main__':
    # nltk.download('stopwords')
    # nltk.download('punkt')

    train = read_train_file()
    test = read_test_file()

    build_model(train, test)



