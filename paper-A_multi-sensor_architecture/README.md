Feijo's Master Degree - Qualifying Exam
===

## How to use it:

Clone the full repo and pay attention for the if the data folder follows the structure
    
    .
    ├── data
    │   ├── csv             - csv concataned and chunks
    │   └── raw             - Raw csv files of each class


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
