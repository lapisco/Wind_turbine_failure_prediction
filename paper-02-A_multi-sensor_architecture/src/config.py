import datetime
import numpy as np
from os.path import join

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

ROOT_DIR = '..'
DATA_FOLDER = join(ROOT_DIR, 'data', 'csv')
MODELS_FOLDER = join(ROOT_DIR, 'models')

RESULTS_FOLDER =  join(ROOT_DIR, 'results')
RESULTS_EMBEDDED_FOLDER = join(RESULTS_FOLDER, 'embedded')
RESULTS_EMBEDDED_RASP = join(RESULTS_EMBEDDED_FOLDER, 'raspberry')
RESULTS_EMBEDDED_JETTSON = join(RESULTS_EMBEDDED_FOLDER, 'jettson')
RESULTS_EMBEDDED_XAVIER = join(RESULTS_EMBEDDED_FOLDER, 'xavier')


PIPELINE_FOLDER = join(MODELS_FOLDER, 'pipeline')
PIPELINE_BASE_NAME = 'tf_pipeline'
SELECTOR_BASE_NAME = 'tf_selector'
SCALER_BASE_NAME = 'tf_scaler'
SELECTOR_FOLDER = join(MODELS_FOLDER, 'selector')
SCALER_FOLDER = join(MODELS_FOLDER, 'scaler')

CLASSIFIER_BASE_NAME = 'clf'
CLASSIFIER_FOLDER = join(MODELS_FOLDER, 'classifier')

MT_RUNS = 2
ROUND = 4

DATA = {
        'SENSORC':
        {
            'FOURIER': 'v000_SCIG_SC_SENSORC_FOURIER',
            'HOS': 'v000_SCIG_SC_SENSORC_HOS',
            'SCM': 'v000_SCIG_SC_SENSORC_SCM',
        },
        'SENSORA':
        {
            'FOURIER': 'v000_SCIG_SC_SENSORA_FOURIER',
            'HOS': 'v000_SCIG_SC_SENSORA_HOS',
            'SCM': 'v000_SCIG_SC_SENSORA_SCM',
        },
        'SENSORV':
        {
            'FOURIER': 'v000_SCIG_SC_SENSORV_FOURIER',
            'HOS': 'v000_SCIG_SC_SENSORV_HOS',
            'SCM': 'v000_SCIG_SC_SENSORV_SCM',
        }        
}



SELECTED_FEATURES = {
    'SENSORC':
                        {
                        'FOURIER':
                         {
                            'SF1':  \
                                ['fx0d5_R', 'fx1d5_R', 'fx2d5_R', 'fx3_R', 'fx5_R', 'fx7_R', 'I_R_rms'],
                            'SF2': \
                                ['fx0d5_S', 'fx1d5_S', 'fx2d5_S', 'fx3_S', 'fx5_S', 'fx7_S', 'I_S_rms'], 
                            'SF3': \
                                ['fx0d5_T', 'fx1d5_T', 'fx2d5_T', 'fx3_T', 'fx5_T', 'fx7_T', 'I_T_rms'], 
                            'SF4': \
                                ['Freq_Gen', 'CC_bus'],  
                        },
                        'HOS': 
                        {
                            'SF1':  \
                                ['Skewness_R', 'Kurtosis_R', 'Variance_R', 'RMS_R'],
                            'SF2': \
                                ['Skewness_S', 'Kurtosis_S', 'Variance_S', 'RMS_S'],
                            'SF3': \
                                ['Skewness_T', 'Kurtosis_T', 'Variance_T', 'RMS_T'],
                            'SF4': \
                                ['Freq_Gen', 'CC_bus'],  
                        },
                        'SCM': 
                        {
                            'SF1':  \
                                ['scm_COR_R', 'scm_IDM_R', 'scm_ENT_R', 'scm_CSD_R', 'scm_CSR_R', 'I_R_rms'],
                            'SF2': \
                                ['scm_COR_S', 'scm_IDM_S', 'scm_ENT_S', 'scm_CSD_S', 'scm_CSR_S', 'I_S_rms'],
                            'SF3': \
                                ['scm_COR_T', 'scm_IDM_T', 'scm_ENT_T', 'scm_CSD_T', 'scm_CSR_T', 'I_T_rms'],
                            'SF4': \
                                ['Freq_Gen', 'CC_bus'],  
                        }
                        },
    'SENSORA':
                    {
                        'FOURIER':
                         {
                            'SF1':  \
                                ['fx1_R','fx0d5_R', 'fx1d5_R', 'fx2d5_R', 'fx3_R', 'fx5_R', 'fx7_R'],
                            'SF2': \
                                ['Freq_Gen', 'CC_bus'],  
                        },
                        'HOS': 
                        {
                            'SF1':  \
                                ['Skewness_R', 'Kurtosis_R', 'Variance_R', 'RMS_R'],
                            'SF2': \
                                ['Freq_Gen', 'CC_bus'],  
                        },
                        'SCM': 
                        {
                            'SF1':  \
                                ['scm_COR_R', 'scm_IDM_R', 'scm_ENT_R', 'scm_CSD_R', 'scm_CSR_R'],
                            'SF2': \
                                ['Freq_Gen', 'CC_bus'],  
                        }
                    },
    'SENSORV':
                    {
                        'FOURIER':
                         {
                            'SF1':  \
                                ['fx0d5_X', 'fx1d5_X', 'fx2d5_X', 'fx3_X', 'fx5_X', 'fx7_X'],
                            'SF2': \
                                ['fx0d5_Y', 'fx1d5_Y', 'fx2d5_Y', 'fx3_Y', 'fx5_Y', 'fx7_Y'], 
                            'SF3': \
                                ['fx0d5_Z', 'fx1d5_Z', 'fx2d5_Z', 'fx3_Z', 'fx5_Z', 'fx7_Z'], 
                            'SF4': \
                                ['Freq_Gen', 'CC_bus'],  
                        },
                        'HOS': 
                        {
                            'SF1':  \
                                ['Skewness_X', 'Kurtosis_X', 'Variance_X', 'RMS_X'],
                            'SF2': \
                                ['Skewness_Y', 'Kurtosis_Y', 'Variance_Y', 'RMS_Y'],
                            'SF3': \
                                ['Skewness_Z', 'Kurtosis_Z', 'Variance_Z', 'RMS_Z'],
                            'SF4': \
                                ['Freq_Gen', 'CC_bus'],  
                        },
                        'SCM': 
                        {
                            'SF1':  \
                                ['scm_COR_X', 'scm_IDM_X', 'scm_ENT_X', 'scm_CSD_X', 'scm_CSR_X'],
                            'SF2': \
                                ['scm_COR_Y', 'scm_IDM_Y', 'scm_ENT_Y', 'scm_CSD_Y', 'scm_CSR_Y'],
                            'SF3': \
                                ['scm_COR_Z', 'scm_IDM_Z', 'scm_ENT_Z', 'scm_CSD_Z', 'scm_CSR_Z'],
                            'SF4': \
                                ['Freq_Gen', 'CC_bus'],  
                        }
                    }
                    }
    
                        
                    
#SEED = np.random.randint(9999)
SEED = 27703389

CLASSIFIERS = {'mlp': MLPClassifier(random_state=SEED), 
'svm': SVC(random_state=SEED), 'knn': KNeighborsClassifier(), 
'naive_bayes': GaussianNB()}

RESULTS = {
    'FOURIER': {
        'mlp':{
            'acc': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'fpr': [],
            'fnr': [],
            'fpr_weighted': [],
            'fnr_weighted': [],
            'test_time_clf': [],
            'test_time_feat': [],
        },
        'svm':{
            'acc': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'fpr': [],
            'fnr': [],
            'fpr_weighted': [],
            'fnr_weighted': [],
            'test_time_clf': [],
            'test_time_feat': [],
        },
        'knn':{
            'acc': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'fpr': [],
            'fnr': [],
            'fpr_weighted': [],
            'fnr_weighted': [],
            'test_time_clf': [],
            'test_time_feat': [],
        },
        'naive_bayes':{
            'acc': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'fpr': [],
            'fnr': [],
            'fpr_weighted': [],
            'fnr_weighted': [],
            'test_time_clf': [],
            'test_time_feat': [],
        }
    },
    'HOS': {
        'mlp':{
            'acc': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'fpr': [],
            'fnr': [],
            'fpr_weighted': [],
            'fnr_weighted': [],
            'test_time_clf': [],
            'test_time_feat': [],
        },
        'svm':{
            'acc': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'fpr': [],
            'fnr': [],
            'fpr_weighted': [],
            'fnr_weighted': [],
            'test_time_clf': [],
            'test_time_feat': [],
        },
        'knn':{
            'acc': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'fpr': [],
            'fnr': [],
            'fpr_weighted': [],
            'fnr_weighted': [],
            'test_time_clf': [],
            'test_time_feat': [],
        },
        'naive_bayes':{
            'acc': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'fpr': [],
            'fnr': [],
            'fpr_weighted': [],
            'fnr_weighted': [],
            'test_time_clf': [],
            'test_time_feat': [],
        }
    },
    'SCM': {
        'mlp':{
            'acc': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'fpr': [],
            'fnr': [],
            'fpr_weighted': [],
            'fnr_weighted': [],
            'test_time_clf': [],
            'test_time_feat': [],
        },
        'svm':{
            'acc': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'fpr': [],
            'fnr': [],
            'fpr_weighted': [],
            'fnr_weighted': [],
            'test_time_clf': [],
            'test_time_feat': [],
        },
        'knn':{
            'acc': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'fpr': [],
            'fnr': [],
            'fpr_weighted': [],
            'fnr_weighted': [],
            'test_time_clf': [],
            'test_time_feat': [],
        },
        'naive_bayes':{
            'acc': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'fpr': [],
            'fnr': [],
            'fpr_weighted': [],
            'fnr_weighted': [],
            'test_time_clf': [],
            'test_time_feat': [],
        }
    },
}

BASE_METRICS = {
    'acc': [],
    'f1': [],
    'precision': [],
    'recall': [],
    'fpr': [],
    'fnr': [],
    'fpr_weighted': [],
    'fnr_weighted': [],
    'test_time_clf': [],
    'test_time_feat': [],
}

RESULTS_STD = {
    'FOURIER': {
        'mlp': BASE_METRICS,
        'svm': BASE_METRICS,
        'knn': BASE_METRICS,
        'naive_bayes': BASE_METRICS,
    },
    'HOS': {
        'mlp': BASE_METRICS,
        'svm': BASE_METRICS,
        'knn': BASE_METRICS,
        'naive_bayes': BASE_METRICS,
    },
    'SCM': {
        'mlp': BASE_METRICS,
        'svm': BASE_METRICS,
        'knn': BASE_METRICS,
        'naive_bayes': BASE_METRICS,
    },
}

RESULTS_conf = {
    'FOURIER': {
        'mlp':{
            'conf': [],
        },
        'svm':{
            'conf': [],
        },
        'knn':{
            'conf': [],
        },
        'naive_bayes':{
            'conf': [],
        },
    },
    'HOS': {
        'mlp':{
            'conf': [],
        },
        'svm':{
            'conf': [],
        },
        'knn':{
            'conf': [],
        },
        'naive_bayes':{
            'conf': [],
        },
    },
    'SCM': {
        'mlp':{
            'conf': [],
        },
        'svm':{
            'conf': [],
        },
        'knn':{
            'conf': [],
        },
        'naive_bayes':{
            'conf': [],
        }
    }
}

METRICS_COLUMN = ['acc', 'f1', 'precision', 'recall', 'fpr', 'fnr', 'fpr_weighted', 'fnr_weighted',
                 'test_time_clf', 'test_time_feat']

INDEX_DF = [
    'FOURIER_mlp', 'HOS_mlp', 'SCM_mlp',
    'FOURIER_svm', 'HOS_svm', 'SCM_svm',
    'FOURIER_knn', 'HOS_knn', 'SCM_knn',
    'FOURIER_naive_bayes', 'HOS_naive_bayes', 'SCM_naive_bayes',    
    ]


DATA_DF = [
    RESULTS['FOURIER']['mlp'], RESULTS['HOS']['mlp'], RESULTS['SCM']['mlp'],
    RESULTS['FOURIER']['svm'], RESULTS['HOS']['svm'], RESULTS['SCM']['svm'],
    RESULTS['FOURIER']['knn'], RESULTS['HOS']['knn'], RESULTS['SCM']['knn'],
    RESULTS['FOURIER']['naive_bayes'], RESULTS['HOS']['naive_bayes'], RESULTS['SCM']['naive_bayes'],
    ]

DATA_DF_STD = [
    RESULTS_STD['FOURIER']['mlp'], RESULTS_STD['HOS']['mlp'], RESULTS_STD['SCM']['mlp'],
    RESULTS_STD['FOURIER']['svm'], RESULTS_STD['HOS']['svm'], RESULTS_STD['SCM']['svm'],
    RESULTS_STD['FOURIER']['knn'], RESULTS_STD['HOS']['knn'], RESULTS_STD['SCM']['knn'],
    RESULTS_STD['FOURIER']['naive_bayes'], RESULTS_STD['HOS']['naive_bayes'], RESULTS_STD['SCM']['naive_bayes'],
    ]

