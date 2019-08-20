
# coding: utf-8

# In[1]:


# Imports of libraries and frameworks:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import metrics

import tensorflow as tf
# from tensorflow import keras
import keras

# Set the random seed for reproducibility

seed = 69
np.random.seed(seed)
tf.set_random_seed(seed)


# Load dataset:

# In[2]:


# filename = '../../datasets/v000_SCIG_SC_SENSORC_FOURIER_010.csv'
filename = 'datasets/v000_SCIG_SC_SENSORC_FOURIER_010.csv'
dataset = pd.read_csv(filename);


# In[3]:


dataset.head()


# Remove unwanted features

# In[4]:


unwanteFeatures = ['idx', 'fx1_R', 'fx1_S', 'fx1_T', 'Freq_Rated', 'Power']
dataset_important_features = dataset.drop(unwanteFeatures, axis=1)

dataset_important_features.head()


# In[5]:


# X are the inputs and y the outputs:

X = dataset_important_features.values[:,:-1]
y = dataset_important_features.values[:,-1]


# ### Preprocessing data with standardization

# In[6]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler();
std_scaler.fit(X)

X = std_scaler.transform(X)


# ## Training with 10-fold Cross Validation, tensorflow and keras

# In[7]:


print(tf.__version__)


# In[8]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, shuffle=True)


# ### Create a keras model

# Create your own MLP

# In[9]:

print(np.unique(y).size)

mlp = keras.Sequential([
    keras.layers.Dense(units=10, input_shape=(X_train.shape[1]), activation='relu'),
    keras.layers.Dense(units=np.unique(y).size, activation='softmax'),
])


# Define, optimzer, loss and mectrics

# In[10]:


mlp.compile(optimizer=keras.optimizers.Adam(epsilon=1e-8, decay=0.01), 
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy'])


# In[ ]:


mlp.fit(X_train, y_train, epochs=2, verbose=1)


# In[ ]:


test_loss, test_acc = mlp.evaluate(X_test, y_test)

print('Test accuracy:', test_acc)


# ## Training with 10-fold Cross Validation

# In[ ]:


# Create an MLP:
from sklearn.neural_network import MLPClassifier

mlpCLf = MLPClassifier(solver='lbfgs', alpha=1e-5, learning_rate='adaptive', learning_rate_init = 0.1,
                    max_iter=1300, momentum=0.3, activation = 'tanh', power_t=0.5,
                    hidden_layer_sizes=(10,))


# In[ ]:


# Create a cross validation object:

from sklearn.model_selection import cross_val_predict

y_pred = cross_val_predict(mlpCLf, X_prepared, y, cv=10)


# In[ ]:


def percentage_confusion_matrix (mat):
    return np.around(100*(mat / mat.sum(axis=1)[:,None]), 2)


# In[ ]:


print(percentage_confusion_matrix(metrics.confusion_matrix(y_pred, y)))
print(metrics.accuracy_score(y_pred, y))


# In[ ]:


mat = metrics.confusion_matrix(y_pred, y)

plt.matshow((mat / mat.sum(axis=1)[:,None]))
plt.show()

