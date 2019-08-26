import numpy as np 
import pandas as pd 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from config import *

'''
The transformations are applied in the dataframe in the following order:
1 - Data cleaning: remove NaN and split into categorical and numerical features
2 - Remove unwanted features in numerical features
3 - Remove unwanted features in categorical features
4 - Feature scaling in numerical features (This one is created by using StandardScaler from sklearn)
5 - Concat two features set
6 - Feature selction (Optional)
7 - Drop NaN that might appear from the scaling tranformation
'''


class FeatureSelector(BaseEstimator, TransformerMixin):
    '''
    Nothing yet
    '''
    def __init__(self, features_names):
        self.features_names = features_names
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.features_names].values


class FeatureAppend(BaseEstimator):
    '''
    Only works for SENSORC and SENSORA
    '''
    def __init__(self, sensor, feature_set):
        self.sensor = sensor
        self.feature_set = feature_set
    
    def fit(self, X, y=None):
        f1 = SELECTED_FEATURES[self.sensor][self.feature_set]['SF1']
        f2 = SELECTED_FEATURES[self.sensor][self.feature_set]['SF2']
        f3 = SELECTED_FEATURES[self.sensor][self.feature_set]['SF3']
        f4 = SELECTED_FEATURES[self.sensor][self.feature_set]['SF4']
        X_out = np.concatenate(
            (
                X[f1].values, 
                X[f2].values,
                X[f3].values
            ),
            axis = 0
        )
        # frequency and cc_bus
        X_2 = np.concatenate((X[f4], X[f4], X[f4]), axis=0)
        X_out = np.concatenate((X_out, X_2), axis=1)
        y = np.concatenate((X['Class'], X['Class'], X['Class']), axis=0)
        self.X_out = X_out
        self.y = y

        return X_out

    def transform(self, X): 
        return X


class DataCleaning(BaseEstimator, TransformerMixin):
    ''' 
    Clean data according the procedures studied in the notebook analyses-02. In short: 
    (i) Drops Nan; (ii) split data in categorical and numerical features; 
    (iii) 1-hot-enconding of categorical features; (iv) Get a unique categorical features
    of an user in a period of 16 weeks; (v)  Get a unique numerical features of an user 
    in a period of 16 weeks; (vi) Average the numerical features in a period of 16-week;
    (vii) contat both feature set;
    -----
    Methods
    ------------------------
    > fit(df)
    Parameters:
    df: dataframe of the dataset, in which the user name must be set as index; 
    -----
    Returns:
    self
    
    > transform(df)

    Parameters:
    - df: dataframe of the dataset, in which the user name must be set as index; 
    -----
    Returns:
    - dict: a dictonary variable of dataframes: {'numerical': DataFrame, 'categorical': DataFrame}
    
    -----------------
    OBS.: fit_transform method is available, inherited from TransformerMixin class.
    '''
    
    def fit(self, df):
        return self

    def transform(self, df):
        # Remove NaN:
        df_clean = df.dropna(how='any', inplace=False)
        return df_clean

class RemoveFeatures(BaseEstimator, TransformerMixin):
    ''' 
    Remove unwanted features from the dataframes;
    -----
    Initialized parameters:
    - features: str or list cointaining the field that ought to be removed. Default: 'week'.

    Methods
    ------------------------
    > fit(df)
    Parameters:
    - dict: a dictonary variable of dataframes: {'numerical': DataFrame, 'categorical': DataFrame}
    -----
    Returns:
    self
    
    > transform(df)

    Parameters:
    - dict: a dictonary variable of dataframes: {'numerical': DataFrame, 'categorical': DataFrame}
    -----
    Returns:
    - dict: a dictonary variable of dataframes: {'numerical': DataFrame, 'categorical': DataFrame}
    -----------------
    OBS.: fit_transform method is available, inherited from TransformerMixin class.
    '''

    def __init__(self, features='week'):
        self.features = features

    def fit(self, df):
        return self

    def transform(self, df):
        
        return {'numerical': df['numerical'].drop(columns=self.features),
                'categorical': df['categorical'].drop(columns=self.features)}

class FeatureScaling(BaseEstimator, TransformerMixin):
    ''' 
    Scale features by standardization;
    -----
    Initialized parameters:
    - type: str cointaining the scaling method. Default: 'std'.
        - 'std': StandardScaler()

    Methods
    ------------------------
    > fit(df)
    Parameters:
    - dict: a dictonary variable of dataframes: {'numerical': DataFrame, 'categorical': DataFrame}
    -----
    Returns:
    self

    Atrributes:
    self._scaler: saved object that sould be used along with the trained model.
    
    > transform(df)

    Parameters:
    - dict: a dictonary variable of dataframes: {'numerical': DataFrame, 'categorical': DataFrame}
    -----
    Returns:
    - dict: a dictonary variable of dataframes: {'numerical': DataFrame, 'categorical': DataFrame}
    -----------------
    OBS.: fit_transform method is available, inherited from TransformerMixin class.
    '''

    def __init__(self, type='std'):
        self.type = type

    def fit(self, X):
        self._scaler = StandardScaler().fit(X)
        return self

    def transform(self, X):
        if self.type == 'std':
            return self._scaler.transform(X)

class MergeFeatures(TransformerMixin):
    ''' 
    Concat the numerical and categorical dataframes into a single one.
    -----

    Methods
    ------------------------
    > fit(df)
    Parameters:
    - dict: a dictonary variable of dataframes: {'numerical': DataFrame, 'categorical': DataFrame}
    -----
    Returns:
    self
    
    > transform(df)

    Parameters:
    - dict: a dictonary variable of dataframes: {'numerical': DataFrame, 'categorical': DataFrame}
    -----
    Returns:
    - dataframe: a daframe with both feature set.
    -----------------
    OBS.: fit_transform method is available, inherited from TransformerMixin class.
    '''

    def fit(self, df):
        return self

    def transform(self, df):
        return pd.concat([df['numerical'], df['categorical']], axis=1)

class DropNaN(TransformerMixin):
    ''' 
    Drop any row from the dataframe that contains a NaN.
    -----

    Methods
    ------------------------
    > fit(df)
    Parameters:
    - df: a dataframe
    -----
    Returns:
    self
    
    > transform(df)

    Parameters:
    - df: a dataframe
    -----
    Returns:
    - dataframe: a daframe withou NaN.
    -----------------
    OBS.: fit_transform method is available, inherited from TransformerMixin class.
    '''

    def fit(self, df):
        return self

    def transform(self, df):
        return df.dropna()

# class FeatureSelection(TransformerMixin):
#     ''' 
#     Select the relevant features.
#     -----
#     Initialized parameters:
#     - features: str or list of str containing the fields the should be kept

#     Atrributes:
#     self.features: feature names.

#     Methods
#     ------------------------
#     > fit(df)
#     Parameters:
#     - df: a dataframe.
#     -----
#     Returns:
#     self

#     > transform(df)

#     Parameters:
#     - df: a dataframe.
#     -----
#     Returns:
#     - df: a dataframe.
#     -----------------
#     OBS.: fit_transform method is available, inherited from TransformerMixin class.
#     '''
#     def __init__(self, extractor='FOURIER', features=None):
#         if not features:
#             self.features = SELECTED_FEATURES[extractor]
#         self.extractor = extractor

#     def fit(self, df):
#         return self

#     def transform(self, df):
#         return df[self.features]

class GetLables(TransformerMixin):
    ''' 
    Get the labels following the user index in the feature dataframe.
    -----
    Methods
    ------------------------
    > fit(df_user, df_features)
    Parameters:
    - df_user: dataframe containing the user's data
    - df_features: dataframe the outta be used as the feature set. It MUST contain
    the user's name as index.
    -----
    Returns:
    self

    > transform(df_user, df_features)
    
    Parameters:
    - df_user: dataframe containing the user's data
    - df_features: dataframe the outta be used as the feature set. It MUST contain
    the user's name as index.
    -----
    Returns:
    - df: a dataframe.
    -----------------
    OBS.: fit_transform method is available, inherited from TransformerMixin class.
    '''

    def fit(self, df):
        return self

    def transform(self, df):
        return df['Class'].values


if __name__ == "__main__":
    import pandas as pd
    from os.path import join
    from config import *
    
    df = pd.read_csv(join(DATA_FOLDER, DATA['SENSORV']['FOURIER']) + '.csv')
    tf = FeatureAppend(sensor='SENSORV', feature_set='FOURIER')
    X = tf.fit(df)

    print(X.shape)