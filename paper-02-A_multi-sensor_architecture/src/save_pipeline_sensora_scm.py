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
    tf_selector_01 = load_selector('tf_selector_SENSORA_SCM_SF1' + '.pkl')
    tf_selector_02 = load_selector('tf_selector_SENSORA_SCM_SF2' + '.pkl')

    # scalers
    tf_scaler_01 = load_scaler('tf_scaler_SENSORA_SCM_SF1' + '.pkl')
    tf_scaler_02 = load_scaler('tf_scaler_SENSORA_SCM_SF2' + '.pkl')

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

    full_pippeline = FeatureUnion(
        [
            ('features_sf1', pipe_01),
            ('features_sf2', pipe_02),
        ]
    )

    return full_pippeline


if __name__ == "__main__":
    sensor = 'SENSORA'
    feature_set = 'SCM'
    df = pd.read_csv(join(DATA_FOLDER, DATA[sensor][feature_set]) + '.csv')
    
    pipe_name = PIPELINE_BASE_NAME + '_{}_{}.pkl'.format(sensor, feature_set)
    logging.info('Creating pipeline: {}'.format(pipe_name))
    full_pipeline = gen_pipe_union()
    X = full_pipeline.transform(df)
    print(X.shape)
    joblib.dump(full_pipeline, join(PIPELINE_FOLDER, pipe_name))
    logging.info('Pipeline saved: {}'.format(pipe_name))



    

    
    