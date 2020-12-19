import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from re import sub
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from nltk import word_tokenize, RegexpTokenizer, TweetTokenizer

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

import os
import argparse

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree



import nltk
nltk.download('stopwords')
nltk.download('punkt')

def remove_URL(text):
    url = RegexpTokenizer(r'https?://\S+|www\.\S+', gaps = True)
    return " ".join(url.tokenize(text))

def stopWords(tweet):  
    stop_words, toker = stopwords.words('english'), TweetTokenizer()
    words_tokens = toker.tokenize(tweet)
    return " ".join([word for word in  words_tokens if not word in stop_words])

def remove_pontucations(text):
    tokenizer_dots = RegexpTokenizer(r'\w+')
    return " ".join(tokenizer_dots.tokenize(text))

def clean(data):
    data.text = data.text.apply(lambda x: x.lower())
    data.text = data.text.apply(lambda x: " ".join(x.split()))
    data.text = data.text.apply(lambda x: sub(r'\d+', '', x))
    data.text = data.text.apply(lambda x: remove_pontucations(x))
    data.text = data.text.apply(lambda x: stopWords(x))
    data.text = data.text.apply(lambda x: x.replace('_', ' '))
    data.text = data.text.apply(lambda x: remove_URL(x))


def metrics(TrueOutput, PredOutput, Classifier):
    return {'Classifier': Classifier,
            'Accuracy': accuracy_score(TrueOutput, PredOutput),
            'Precision': precision_score(TrueOutput, PredOutput, average='micro'),
            'Recall': recall_score(TrueOutput, PredOutput, average='micro'),
            'F1': f1_score(TrueOutput, PredOutput, average='micro')}


if __name__ == "__main__":
    ##################################################################################################################################
    # Input dos parâmetros do código
    ##################################################################################################################################
    parser = argparse.ArgumentParser(description='Amazon Pets Review')
    parser.add_argument('--max_features', type=int, help='Max_features for CountVectorizer', required = True)
    args = parser.parse_args()
    max_len = args.max_features

    ##################################################################################################################################
    # Criação das pastas de teste
    ##################################################################################################################################
    ClassifierList = ['MLP', 'SVC', 'KNN', 'DT']

    for classifier in ClassifierList:
        if not(os.path.isdir(os.path.join('Resultados', 'CountVectorizer_' + str(max_len), classifier))):
            os.makedirs(os.path.join('Resultados', 'CountVectorizer_' + str(max_len), classifier))

    ##################################################################################################################################
    # Carrega os datasets (Treino, Validação e Teste)
    ##################################################################################################################################
    path = 'amazon-pet-product-reviews-classification'
    data_train = pd.read_csv(os.path.join(path, 'train.csv'))
    val_df = pd.read_csv(os.path.join(path, 'valid.csv'))
    test = pd.read_csv(os.path.join(path, 'test.csv'))

    ##################################################################################################################################
    # Limpa os datasets
    ##################################################################################################################################
    clean(data_train)
    clean(test)
    clean(val_df)

    ##################################################################################################################################
    # Seleciona os Inputs e Outputs do (Treino, Validação e Teste)
    ##################################################################################################################################
    X_train, X_val, y_train, y_val, X_test = data_train.text, val_df.text, data_train.label, val_df.label, test.text
    
    ##################################################################################################################################
    # Vetoriza as strings de input
    ##################################################################################################################################
    vect = CountVectorizer(analyzer='word', stop_words='english', tokenizer=word_tokenize, max_features=max_len)
    X_train = vect.fit_transform(X_train)
    X_val = vect.transform(X_val)
    X_test = vect.transform(X_test)

    ##################################################################################################################################
    # Inicio do teste com os classificadores
    ##################################################################################################################################
    Metrics = []

    ##################################################################################################################################
    # MultiLayer Perceptron
    ##################################################################################################################################
    print('Processing MLP')
    clf = MLPClassifier()
    clf = clf.fit(X_train, y_train)
    predict = clf.predict(X_val)

    Metrics.append(metrics(y_val, predict, 'MLP'))

    f = open(os.path.join('Resultados', 'CountVectorizer_' + str(max_len), 'MLP', 'Confusion_Matrix.txt'), "w")
    f.write(str(confusion_matrix(y_val, predict)))
    f.close()

    y_preds = clf.predict(X_test)
    test['label'] = y_preds
    submission = test[['id', 'label']]
    submission.to_csv(os.path.join('Resultados', 'CountVectorizer_' + str(max_len), 'MLP', 'submission.csv'), index=False)

    ##################################################################################################################################
    # Support Vector Classifier
    ##################################################################################################################################
    print('Processing SVC')
    clf = SVC()
    clf = clf.fit(X_train, y_train)
    predict = clf.predict(X_val)

    Metrics.append(metrics(y_val, predict, 'SVC'))

    f = open(os.path.join('Resultados', 'CountVectorizer_' + str(max_len), 'SVC', 'Confusion_Matrix.txt'), "w")
    f.write(str(confusion_matrix(y_val, predict)))
    f.close()

    y_preds = clf.predict(X_test)
    test['label'] = y_preds
    submission = test[['id', 'label']]
    submission.to_csv(os.path.join('Resultados', 'CountVectorizer_' + str(max_len), 'SVC', 'submission.csv'), index=False)

    ##################################################################################################################################
    # K Nearest Neighbors Classifier
    ##################################################################################################################################
    print('Processing KNN')
    clf = KNeighborsClassifier()
    clf = clf.fit(X_train, y_train)
    predict = clf.predict(X_val)

    Metrics.append(metrics(y_val, predict, 'KNN'))

    f = open(os.path.join('Resultados', 'CountVectorizer_' + str(max_len), 'KNN', 'Confusion_Matrix.txt'), "w")
    f.write(str(confusion_matrix(y_val, predict)))
    f.close()

    y_preds = clf.predict(X_test)
    test['label'] = y_preds
    submission = test[['id', 'label']]
    submission.to_csv(os.path.join('Resultados', 'CountVectorizer_' + str(max_len), 'KNN', 'submission.csv'), index=False)


    ##################################################################################################################################
    # Decision Tree Classifier
    ##################################################################################################################################
    print('Processing DT')
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    predict = clf.predict(X_val)

    Metrics.append(metrics(y_val, predict, 'DT'))

    f = open(os.path.join('Resultados', 'CountVectorizer_' + str(max_len), 'DT', 'Confusion_Matrix.txt'), "w")
    f.write(str(confusion_matrix(y_val, predict)))
    f.close()

    y_preds = clf.predict(X_test)
    test['label'] = y_preds
    submission = test[['id', 'label']]
    submission.to_csv(os.path.join('Resultados', 'CountVectorizer_' + str(max_len), 'DT', 'submission.csv'), index=False)


    ##################################################################################################################################
    # Exporta os resultados finais
    ##################################################################################################################################

    Metrics = pd.DataFrame.from_dict(Metrics)

    Metrics.to_csv(os.path.join('Resultados', 'CountVectorizer_' + str(max_len), 'FullMetrics.csv'), index=False)
    print(Metrics)