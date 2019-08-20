from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
import pandas as pd
import os
import sys
import logging
logging.basicConfig(level=logging.INFO)

from transforms import *
from config import *

for sensor in DATA.keys():
    logging.info('=================')
    for feature_set in DATA[sensor].keys():
        for features in SELECTED_FEATURES[sensor][feature_set].keys():
            logging.info('Selected sensor: {}'.format(sensor))
            logging.info('Selected feature_set: {}'.format(feature_set))
            logging.info('Features are: {}'.format(features))
            logging.info('{}'.format(SELECTED_FEATURES[sensor][feature_set][features]))
            
            dataset = DATA[sensor][feature_set] + '.csv'
            features_names = SELECTED_FEATURES[sensor][feature_set][features]
    
            logging.info('Loading data from: {}'.format(dataset))
            df = pd.read_csv(os.path.join(DATA_FOLDER, DATA[sensor][feature_set]) + '.csv')
            
            # Selector
            selector = FeatureSelector(features_names=features_names)
            selector_name = SELECTOR_BASE_NAME + '_{}_{}_{}.pkl'.format(sensor, feature_set, features)
            data = selector.fit_transform(df)
            joblib.dump(selector, os.path.join(SELECTOR_FOLDER, selector_name))
            logging.info('Selector saved: {}'.format(selector_name))

            # Scaler
            logging.info('Scaling data...')
            scaler = FeatureScaling()
            scaler_name = SCALER_BASE_NAME + '_{}_{}_{}.pkl'.format(sensor, feature_set, features)
            scaler.fit_transform(data)
            joblib.dump(scaler, os.path.join(SCALER_FOLDER, scaler_name))
            logging.info('Scaler saved: {}'.format(scaler_name))

            # pipeline = Pipeline('selector', FeatureSelection(extractor=feature_set)),
            #                 ('scaler', FeatureScaling()),
            #                 ])
        # _ = pipeline.fit_transform(data)

        # pipe_name = PIPELINE_BASE_NAME + '_{}_{}.pkl'.format(sensor, feature_set)
        # joblib.dump(pipeline, os.path.join(MODELS_FOLDER, pipe_name))
        # logging.info('Pipeline saved: {}'.format(pipe_name))






































