import pandas as pd
import os

DATAFOLDER = os.path.join('..', 'data', 'raw', 'Vibration')
FEATURE_EXTRACTORS = ['FOURIER', 'HOS', 'SCM']
FILENAME = 'v000_SCIG_SC_SENSORV'


def check_extractor(file):
    for feat in FEATURE_EXTRACTORS:
        if feat in file:
            return feat
    raise Exception('Not feature extractor found in theses datasets')
    

def read_csv(file):
    # open with pandas
    return pd.read_csv(file, index_col='idx')


def concat_data(data):
    dataframe = pd.DataFrame([])
    for dataset in data:
        dataframe = pd.concat([dataframe, dataset])


def save_data(data, filename, folder_to_save=os.path.join(DATAFOLDER, 'csv')):
    if not os.path.exists(folder_to_save):
        os.mkdir(folder_to_save)   
    
    data.to_csv(os.path.join(folder_to_save, filename))
    
    pass
    

# 0: FOURIER | 1: HOS | 2: SCM | 3: GOERTZEL
data_featureExtractor = {'FOURIER': [], 'HOS': [], 'SCM': []}
data_class = []

for _, dirs, _ in os.walk(DATAFOLDER):
    for directory in dirs:
        print('Directory: {}'.format(directory))
        for root, _, files in os.walk(os.path.join(DATAFOLDER, directory)):
            for file in files:
                if file.endswith('.csv'):
                    print('Reading file: {}'.format(file))
                    feature_set = check_extractor(os.path.join(root, file))
                    data_featureExtractor[feature_set].append(read_csv(os.path.join(root, file)))
        
        # Finished this directory, i.e., class
        # data_class.append(data_featureExtractor)

# concat all subclasses's data:
print('Concatenating classes...')  
for feature_set in data_featureExtractor.keys():
    print(feature_set)
    data_to_save = pd.concat([pd.DataFrame(d) for d in data_featureExtractor[feature_set]])
    filename = '{}_{}.csv'.format(FILENAME, feature_set)
    print(filename)
    # filename = '{}_{}.csv'.format(FILE, feature_set)
    save_data(data_to_save, filename, folder_to_save=os.path.join('..', 'data', 'csv'))
                    
