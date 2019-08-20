import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from os.path import join
import os
from sklearn.externals import joblib
from config import *


def check_folder(folder):
   if not os.path.exists(folder):
      os.system('mkdir {}'.format(folder))


def load_selector(tf_name):
    return joblib.load(join(SELECTOR_FOLDER, tf_name))

def load_scaler(tf_name):
    return joblib.load(join(SCALER_FOLDER, tf_name))
    

def fpr_fnr_score(conf):
   conf = percentage_confusion_matrix(conf)
   # print(conf)
   tn = conf[0,0]
   fp = conf[0,1:].sum()/conf[0,1:].shape[0]
   fn = conf[1:,0].sum()/conf[0,1:].shape[0]
   tp = conf[1:,1:].sum()/conf[0,1:].shape[0]

   fp_low = conf[0,1:4]
   fp_high = conf[0,4:]

   fn_low = conf[1:4,0]
   fn_high = conf[4:,0]

   fn_weighted = (((1*fn_low[0] + 1.5*fn_low[1] + 2.5*fn_low[2])/5) + \
               ((1*fn_high[0] + 1.5*fn_high[1] + 2.5*fn_high[2])/5))/2

   fp_weighted = (((1*fp_low[0] + 1.5*fp_low[1] + 2.5*fp_low[2])/5) + \
               ((1*fp_high[0] + 1.5*fp_high[1] + 2.5*fp_high[2])/5))/2

   return {'fpr': fp, 'fnr': fn, 'fpr_weighted': fp_weighted, 'fnr_weighted': fn_weighted}
   


def percentage_confusion_matrix(confMat):
    return np.around((confMat / np.sum(confMat,axis=1)[:,None])*100,2)


def print_metrics(y_true, y_hat):
   conf_mat = confusion_matrix(y_true, y_hat)
   acc = accuracy_score(y_true, y_hat)

   print('Confusion Matrix:')
   print(percentage_confusion_matrix(conf_mat))
   print('Accuracy: {}'.format(acc))
   


def save_dict(d, folder):
   for feat in d.keys():
      for clf in d[feat].keys():
         df = pd.DataFrame(d[feat][clf]['conf'])
         df.to_csv(join(folder, 'conf_' + feat + '_' + clf + '.csv'))
         
