A multi-sensor architecture for robust identificationof incipient short-circuits in wind turbine generators
===

The source code necessary to run all experiments discussed in the paper are available here.

## Repo structure

Clone the full repo and pay attention for the if the data folder follows the structure
    
    .
    ├── data
    │   ├── csv             - csv concataned and chunks
    ├── models              - models, transformers and pippelines
    ├── notebooks           - notebooks for data analyses
    ├── results             - results folder
    └── src                 - source code


## Data folder
Inside the csv folder there are the sensor settings datasets in a `csv` format ready to be fed into the model's pipeline. The naming convention refer to parameters, for example, the file `v000_SCIG_SC_SENSORC_FOURIER.csv` tell us,


- `v000`: version of the database (concerns only to author).
- `SCIG`: electrical machine is a Squirrel-Cage Induction Generator (SCIG).
- `SC`: The inserted fault is short-circuit (SC)
- `SENSORC`: The sensor used to monitor the machine. 
- `FOURIER`: The feature extracted method used to build the dataset.

## Models folder

It contains all instaces used in the simulations. The objects are serialized in a pickle format using the extension `.pkl`. Unpackle it using the proper method. The libraty `joblib` is recommended. This folder is organized as follows

- [classifier](classifier/): contain the trainned classifiers. They follow a straightfoward naming convention. For example, `clf_SENSORC_FOURIER_mlp_union.pkl` stands for the MLP classifiers, using the feature union process, trained on the dataset built using the feature extraction method Fourier and the raw signal is a three-phases current sensor.
- [pipeline](pipeline/): contains the transformer that ought to be used to prepare inputs for the classifiers. The process is decribed in paper's Section II.A. The pipeline is a composition of scalers and selector.
- [scaler](scaler): The feature scaler used to standandize features.
- [selector](selector): The feature selector to be applied in the datasets. The naming `SF` stands for set of feature and the labeling `1` or `2` refers to the set of features and the extra features, respectivelly. Describe in paper's Section II.B.


## How to run

1. Navigate to [src](src/) and setup the environment:

```shell
cd src
virtualenv -p python3.6 .env
source .env/bin/activate
pip install -r requirements.txt
```

2. Run the following script for a **single test**

- `python test_clf.py` to print all metrics from the classifiers.


3. Run the whole experinet script for a single test

**INSIDE THE RASPBERRY PI OR JETSSON RUN THE FOLLOWING** 
- `python run_experiment.py` to save all results under the [results dir](src/results/)

You could set the parameters:
- `--runs`: number of independent simulations (*default: 10*);
- `--round`: approximate number to the follow (*default: 4*);
- `--output`: folder wherein the results ought to be stored (*default: under ../results*);
- `--seed`: seed to set for the simulation (*default: np.random state*);

Example: `python run_experiment.py --runs 50`


## More info:

The trained models and pipeline for each feature extractor is saved under [the models dir](models/).

General descriptions about the scripts under the [src dir](src/):

- `dataset_concat.py` concatanet the subset of csv unde the [data/raw/](data/raw/).
- `example-01.py` example script how to use the classes under the [transform.py](src/transform.py).
- `save_pipeline.py` is used to save the pipelines for each feature extractor, under the [models dir](models/)
- `train_clf.py` is used to train all the clasifiers.
- `transforms.py` are the transformers classes to build a pipeline, following the scikit-learn structure.
- `utils.py` general utilities stuff.
