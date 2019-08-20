from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import os
import sys
import logging
filename_log = '.experiment.log'
if os.path.exists(os.path.join(os.path.dirname(__file__), filename_log)):
    os.remove(os.path.join(os.path.dirname(__file__), filename_log))
logging.basicConfig(level=logging.DEBUG, filename=filename_log)

import time
import argparse

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from utils import *
from config import *
import transforms



# clf = {'mlp': [], 'svm': [], 'knn': [], 'naive_bayes': []}

SENSOR = ['SENSORA', 'SENSORC', 'SENSORV']

def run(args):
    for sensor in SENSOR:
        args['output'] = os.path.join(RESULTS_EMBEDDED_FOLDER, args['system'],  sensor, 'appended')
        print(args['output'])
        for feature_set in DATA[sensor].keys():
            logging.debug('************************************************')
            logging.debug('Feature set: {}'.format(feature_set))
            data = pd.read_csv(os.path.join(DATA_FOLDER, DATA[sensor][feature_set]) + '.csv')

            pipe_name = os.path.join(PIPELINE_FOLDER,
                PIPELINE_BASE_NAME + '_{}_{}_appended.pkl'.format(sensor, feature_set))    
            logging.debug('Loading pipeline: {}'.format(pipe_name))
            pipeline = joblib.load(pipe_name)
            if sensor is 'SENSORC' or sensor is 'SENSORV':
                logging.debug('Transforming data')
                tf_feature_append = transforms.FeatureAppend(sensor=sensor, feature_set=feature_set)
                X = tf_feature_append.fit(data)
                start_feat = time.time()
                X = pipeline.transform(X)
                print("Shape of dataset: {}".format(X.shape))
                end_feat = time.time()
                y = tf_feature_append.y
            else:
                start_feat = time.time()
                X = pipeline.transform(data)
                end_feat = time.time()
                y = transforms.GetLables().fit_transform(data)
            
            skf = StratifiedKFold(n_splits=args['runs'], random_state=args['seed'], shuffle=True)
            
            for classifier in CLASSIFIERS.keys():
                logging.debug('Loading classifier: {}'.format(classifier))
                clf_name = CLASSIFIER_BASE_NAME + '_{}_{}_{}_appended.pkl'.format(sensor, feature_set, classifier)
                logging.debug('Loading classifier: {}'.format(clf_name))
                clf = joblib.load(join(CLASSIFIER_FOLDER, clf_name))
                logging.debug('Classifiers loaded')
                
                it = 0
                conf = np.zeros((7,7))
                acc = []
                f1 = []
                recall = []
                precision = []
                fpr = []
                fnr = []
                fpr_weighted = []
                fnr_weighted = []
                time_array = []
                time_array_feat = []

                for train_index, test_index in skf.split(X, y):
                    it += 1
                    X_train = X[train_index]
                    X_test = X[test_index]
                    y_train = y[train_index]
                    y_test = y[test_index]
                    
                    time_array_feat.append(end_feat - start_feat)

                    logging.debug('======================')
                    logging.debug('Monte Carlo run: {}'.format(it))

                    logging.debug('Predicting with classifier: {}'.format(classifier))
                    # clf.fit(X_train, y_train)
                    start = time.time()
                    y_hat = clf.predict(X_test)
                    end = time.time()
                    time_array.append(end - start)

                    logging.debug('Calculating metrics... {}'.format(classifier))
                    # print_metrics(y_test, y_hat)
                    acc.append(accuracy_score(y_test, y_hat))
                    f1.append(f1_score(y_test, y_hat, average='weighted'))
                    precision.append(precision_score(y_test, y_hat, average='weighted'))
                    recall.append(recall_score(y_test, y_hat, average='weighted'))
                    # conf += percentage_confusion_matrix(confusion_matrix(y_test, y_hat))/2
                    conf += confusion_matrix(y_test, y_hat)

                    # Rates
                    rates = fpr_fnr_score(conf)
                    fpr.append(rates['fpr'])
                    fnr.append(rates['fnr'])
                    fpr_weighted.append(rates['fpr_weighted'])
                    fnr_weighted.append(rates['fnr_weighted'])


                logging.debug('{} results:'.format(classifier))
                logging.debug('ACC \n {}'.format(acc))
                logging.debug('Confusion matrix \n {}'.format(percentage_confusion_matrix(conf/MT_RUNS)))

                # Dumping results
                check_folder(os.path.join(args['output'], 'pkl'))
                joblib.dump(acc, os.path.join(args['output'], 'pkl', '{}_{}_acc.pkl'.format(feature_set, classifier)))
                joblib.dump(fpr, os.path.join(args['output'], 'pkl', '{}_{}_fpr.pkl'.format(feature_set, classifier)))
                joblib.dump(fnr, os.path.join(args['output'], 'pkl', '{}_{}_fnr.pkl'.format(feature_set, classifier)))
                joblib.dump(fpr_weighted, os.path.join(args['output'], 'pkl', '{}_{}_fpr_weighted.pkl'.format(feature_set, classifier)))
                joblib.dump(fnr_weighted, os.path.join(args['output'], 'pkl', '{}_{}_fnr_weighted.pkl'.format(feature_set, classifier)))


                RESULTS[feature_set][classifier]['acc'] = 100*np.mean(acc)
                RESULTS[feature_set][classifier]['f1'] = 100*np.mean(f1)
                RESULTS[feature_set][classifier]['precision'] = 100*np.mean(precision)
                RESULTS[feature_set][classifier]['recall'] = 100*np.mean(recall)
                RESULTS[feature_set][classifier]['fpr'] = np.mean(fpr)
                RESULTS[feature_set][classifier]['fnr'] = np.mean(fnr)
                RESULTS[feature_set][classifier]['fpr_weighted'] = np.mean(fpr_weighted)
                RESULTS[feature_set][classifier]['fnr_weighted'] = np.mean(fnr_weighted)
                RESULTS[feature_set][classifier]['test_time_clf'] = 1000*np.mean(time_array)
                RESULTS[feature_set][classifier]['test_time_feat'] = 1000*np.mean(time_array_feat)
                

                RESULTS_STD[feature_set][classifier]['acc'] = 100*np.std(acc)
                RESULTS_STD[feature_set][classifier]['f1'] = 100*np.std(f1)
                RESULTS_STD[feature_set][classifier]['precision'] = 100*np.std(precision)
                RESULTS_STD[feature_set][classifier]['recall'] = 100*np.std(recall)
                RESULTS_STD[feature_set][classifier]['fpr'] = np.std(fpr)
                RESULTS_STD[feature_set][classifier]['fnr'] = np.std(fnr)
                RESULTS_STD[feature_set][classifier]['fpr_weighted'] = np.std(fpr_weighted)
                RESULTS_STD[feature_set][classifier]['fnr_weighted'] = np.std(fnr_weighted)
                RESULTS_STD[feature_set][classifier]['test_time_clf'] = 1000*np.std(time_array)
                RESULTS_STD[feature_set][classifier]['test_time_feat'] = 1000*np.std(time_array_feat)

                RESULTS_conf[feature_set][classifier]['conf'] = percentage_confusion_matrix(conf/MT_RUNS)

        # Save to dataframe

        df_results = pd.DataFrame(
            index=INDEX_DF,
            columns=METRICS_COLUMN,
            data = DATA_DF,
        )


        df_results_std = pd.DataFrame(
            index=INDEX_DF,
            columns=METRICS_COLUMN,
            data = DATA_DF_STD,
        )

        print(df_results.round(args['round']))


        df_results.round(args['round']).to_csv(os.path.join(args['output'], 'results_acc.csv'))
        df_results_std.round(args['round']).to_csv(os.path.join(args['output'], 'results_std.csv'))

        save_dict(RESULTS_conf, args['output'])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', default=MT_RUNS, type=int)
    parser.add_argument('--round', default=ROUND, type=int)
    # parser.add_argument('--output', default=os.path.join(RESULTS_FOLDER, 'SENSORC', 'union'), type=str)
    parser.add_argument('--seed', default=SEED, type=str)
    parser.add_argument('--system', required=False, help='set it to "raspberry", "jettson" or "xavier".')

    args = vars(parser.parse_args())    

    # args['system'] = 'raspberry'

    run(args)