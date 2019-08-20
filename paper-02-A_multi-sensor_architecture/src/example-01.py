import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from transforms import *

import os

DATA_FOLDER = '../data/csv'
DATA_NAME = 'v000_SCIG_SC_SENSORA_SCM_chunk_10.csv'
LABELS = 'v000_SCIG_SC_SENSORA_SCM_labels_chunk_10.csv'

features = pd.read_csv(os.path.join(DATA_FOLDER, DATA_NAME))
labels = pd.read_csv(os.path.join(DATA_FOLDER, DATA_NAME))


print(features.head())

# Data Cleaning
tf_cleaner = DropNaN()
df = tf_cleaner.fit_transform(features)
# Feature Selection
tf_selector = FeatureSelection(extractor='SCM')
df = tf_selector.fit_transform(df)
# Data Scaling
tf_scaler = FeatureScaling()
df = tf_scaler.fit_transform(df)

pipeline = Pipeline([('cleaner', tf_cleaner),
                     ('selector', tf_selector),
                     ('scaler', tf_scaler),
                    ])

print(df.head())