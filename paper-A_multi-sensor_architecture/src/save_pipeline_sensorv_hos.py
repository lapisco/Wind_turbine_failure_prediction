from sklearn.externals import joblib
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

from config import *
from utils import *
from transforms import *


def gen_pipe_union():
    tf_selector_01 = load_selector('tf_selector_SENSORV_HOS_SF1' + '.pkl')
    tf_selector_02 = load_selector('tf_selector_SENSORV_HOS_SF2' + '.pkl')
    tf_selector_03 = load_selector('tf_selector_SENSORV_HOS_SF3' + '.pkl')
    tf_selector_04 = load_selector('tf_selector_SENSORV_HOS_SF4' + '.pkl')

    # scalers
    tf_scaler_01 = load_scaler('tf_scaler_SENSORV_HOS_SF1' + '.pkl')
    tf_scaler_02 = load_scaler('tf_scaler_SENSORV_HOS_SF2' + '.pkl')
    tf_scaler_03 = load_scaler('tf_scaler_SENSORV_HOS_SF3' + '.pkl')
    tf_scaler_04 = load_scaler('tf_scaler_SENSORV_HOS_SF4' + '.pkl')

    # pipeline 01: Feature Union
    pipe_01 = Pipeline(
        [
            ('selector_sf1', tf_selector_01),
            ('scaler_sf1', tf_scaler_01),
        ]
    )

    pipe_02 = Pipeline(
        [
            ('selector_sf2', tf_selector_02),
            ('scaler_sf2', tf_scaler_02),
        ]
    )

    pipe_03 = Pipeline(
        [
            ('selector_sf3', tf_selector_03),
            ('scaler_sf3', tf_scaler_03),
        ]
    )

    pipe_04 = Pipeline(
        [
            ('selector_sf4', tf_selector_04),
            ('scaler_sf4', tf_scaler_04),
        ]
    )

    full_pippeline = FeatureUnion(
        [
            ('features_sf1', pipe_01),
            ('features_sf2', pipe_02),
            ('features_sf3', pipe_03),
            ('features_sf4', pipe_04),
        ]
    )

    return full_pippeline


def gen_pipe_single(X):
    scaler = StandardScaler()
    full_pippeline = Pipeline(
        [
            ('scaler', scaler),
        ]
    )

    return full_pippeline

if __name__ == "__main__":
    sensor = 'SENSORV'
    feature_set = 'HOS'
    df = pd.read_csv(join(DATA_FOLDER, DATA[sensor][feature_set]) + '.csv')
    
    pipe_name = PIPELINE_BASE_NAME + '_{}_{}_union.pkl'.format(sensor, feature_set)
    logging.info('Creating pipeline: {}'.format(pipe_name))
    full_pipeline = gen_pipe_union()
    X = full_pipeline.transform(df)
    print(X.shape)
    joblib.dump(full_pipeline, join(PIPELINE_FOLDER, pipe_name))
    logging.info('Pipeline saved: {}'.format(pipe_name))

    ###

    pipe_name = PIPELINE_BASE_NAME + '_{}_{}_appended.pkl'.format(sensor, feature_set)
    logging.info('Creating pipeline: {}'.format(pipe_name))
    tf_feature_append = FeatureAppend(sensor=sensor, feature_set=feature_set)
    X = tf_feature_append.fit(df)

    full_pipeline = gen_pipe_single(X)
    X = full_pipeline.fit_transform(X)
    print(X.shape)

    joblib.dump(full_pipeline, join(PIPELINE_FOLDER, pipe_name))
    logging.info('Pipeline saved: {}'.format(pipe_name))
    



    

    
    