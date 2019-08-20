from sklearn.externals import joblib
import pandas as pd
import os
import sys
import logging
logging.basicConfig(level=logging.INFO)

# clf specs
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


DATA_FOLDER = '../data/csv'
MODELS_FOLDER = '../models'
PIPELINE_BASE_NAME = 'transformer_pipeline'
CLASSIFIER_BASE_NAME = 'clf'

DATA = {'FOURIER': 'v000_SCIG_SC_SENSORC_FOURIER_010_chunk_90'}
LABELS = {'FOURIER': 'v000_SCIG_SC_SENSORC_FOURIER_010_labels_chunk_90'}


clf = {'mlp': [], 'svm': [], 'knn': [], 'naive_bayes': []}

clf['mlp'] = MLPClassifier(activation='tanh', hidden_layer_sizes=(100,), 
                            learning_rate='adaptive', tol=1e-6, max_iter=2000, random_state=SEED)
clf['svm'] = SVC(random_state=SEED)
clf['knn'] = KNeighborsClassifier()
clf['naive_bayes'] = GaussianNB()

for feature_set in DATA.keys():
    logging.info('Feature set: {}'.format(feature_set))
    data = pd.read_csv(os.path.join(DATA_FOLDER, DATA[feature_set]) + '.csv')
    labels = pd.read_csv(os.path.join(DATA_FOLDER, LABELS[feature_set]) + '.csv')
    pipeline = joblib.load(os.path.join(MODELS_FOLDER, PIPELINE_BASE_NAME + '_{}.pkl'.format(feature_set)))

    features = pipeline.transform(data)
    logging.info(features.head())

    X = features.values
    y = labels.values.reshape(-1)

    for classifier in clf.keys():
        logging.info('Classifier: {}'.format(classifier))

        clf[classifier].fit(X, y)

        clf_name = CLASSIFIER_BASE_NAME + '_{}_{}.pkl'.format(feature_set, classifier)
        joblib.dump(clf[classifier], os.path.join(MODELS_FOLDER, clf_name))
        logging.info('Classifier {} saved as: {}'.format(classifier, clf_name))