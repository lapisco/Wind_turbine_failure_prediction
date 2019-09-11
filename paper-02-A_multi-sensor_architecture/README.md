A multi-sensor architecture for robust identification of incipient short-circuits in wind turbine generators
===

The source code necessary to run all experiments discussed in the paper.

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

All datasets used in the paper are inside this folder. [More details about theses dataset are here.](data/csv/)

## Models folder

It contains all instaces used in the simulations. The objects are serialized in a pickle format using the extension `.pkl`. Unpackle it using the proper method. The libraty `joblib` is recommended. This folder is organized as follows

- [classifier](classifier/): contain the trainned classifiers. They follow a straightfoward naming convention. For example, `clf_SENSORC_FOURIER_mlp_union.pkl` stands for the MLP classifiers, using the feature union process, trained on the dataset built using the feature extraction method Fourier and the raw signal is a three-phases current sensor.
- [pipeline](pipeline/): contains the transformer that ought to be used to prepare inputs for the classifiers. The process is decribed in paper's Section II.A. The pipeline is a composition of scalers and selector.
- [scaler](scaler): The feature scaler used to standandize features.
- [selector](selector): The feature selector to be applied in the datasets. The naming `SF` stands for set of feature and the labeling `1` or `2` refers to the set of features and the extra features, respectivelly. Describe in paper's Section II.B.


## How to run

**It is very important to maintain this repository structure. Instabilities may occur if files are moved without caution. Open a question if ran into any problem.**

### Reproductible results

1. Navigate to [src](src/) and setup the environment:

```shell
cd src
virtualenv -p python3.6 .venv
source .venv/bin/activate
pip install -r requirements.txt
```
> You can uso `pipenv` instead of virtualenv, there is a `Pipfile` inside the folder.

2. Run the following scripts to reproduce **our results**

- `python test_clf_appended.py` and `python test_clf_union.py`. The results are saved under the folder [results](results/).

<!-- - `python run_experiment_union.py` and `python run_experiment_appended.py`. The results are saved under the folder [results](results/). -->

> See the [src directory](src/) for more details about the scripts.using another scritps. There is **a lot more** that can be done. And the [results folder](results/) for more details how the results are organized.

### Statistical tests

The statistical tests are made under the [notebook dir](notebooks/). To reproducetheses results open the `notebook-02-statistical_test.ipynb`. The discussion and methods are pretty straightforward.

> To run this parte ot is not necessary to run the aforementioned scripts. The results are store in `.pkl`files [here](results). 



## More info:

Navigate to each folder for more information. There should be a readme file explain in details all proccess. Fell free to contact us, open questions, report bug or PR. Please remember to cite us :)