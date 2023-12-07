# Data Science with PyTorch

## Contents

- [Data Science with PyTorch](#data-science-with-pytorch)
  - [Contents](#contents)
  - [Introduction](#introduction)
  - [Usage](#usage)
    - [PyTroch Model Construction](#pytroch-model-construction)
    - [PyTorch Model Training and Validation](#pytorch-model-training-and-validation)
    - [PyTorch Model Hyperparameter Tuning](#pytorch-model-hyperparameter-tuning)
    - [Usage Example in Notebook](#usage-example-in-notebook)
  - [Development](#development)

## Introduction

With the release of PyTorch 2.0, more and more DL models are built based on PyTorch. Because of its flexibilty during model construction and its light-weight in package size, it beccomes a more and more popular tool in DL study and production.

This repo is a code-based collection, including, 

* PyTorch model construction.
* PyTorch model training and validation.
* PyTorch model hyperparameter tuning.

which can help beginers to use PyTorch models in study and production.

## Usage

### PyTroch Model Construction

In the module `pytorch_model_data_science.model`, there are several pre-defined model structures, including, 

* Regressor
  * DNNNetRegressor
* Classifier
  * DNNNetClassifier

which can be invoked from there directly when necessary.

### PyTorch Model Training and Validation

In the modules `pytorch_model_data_science.train` and `pytorch_model_data_science.validation`, there are functions used for model training and validation.

### PyTorch Model Hyperparameter Tuning

In the modules `pytorch_model_data_science.tune`, there are trainable functions `tune_classifier` and `tune_regressor`, which can be used in hyperparameter tuning for classification or regression models.

### Usage Example in Notebook

Here is a general example about how to use this repo in a realistic data science project.

1. Collect and clean the raw data.
2. Transform categorical features.
3. Normalize features.
4. Convert data to PyTorch Dataset.
5. Run the prepared notebook by the commandline in the notebook, 
   ```
   %run PATH_TO/pytorch_model_data_science/pytorch_model_data_science/tuning.ipynb
   ```
   where there are pre-defined models and trainable functions for hyperparameter tuning. 
6. Tune hyperparameters by using the class `ray.tune.Tuner`, 
   ```python
   # define tuner
   tuner = tune.Tuner(
       trainable=functools.partial(
           trainable_functions,
           ..., # its other arguments except the grid configuration
       ),
       param_space=..., # parameter grid for seaching
       tune_config=tune.tune_config.TuneConfig(
           ..., # tuning configuration
       ),
       run_config=ray.air.config.RunConfig(
           ..., # experiment configuration
       ),
   )

   # tune hyperparameters
   results = tuner.fit()

   # best hyperparameters
   best_result = results.get_best_result()
   best_config = best_result.metrics["config"]
   ```

## Development

* Check and reformat coding format

    ```
    make coding_standards
    ```

* Install requirements

    ```
    poetry install
    ```

* Add dependencies

    ```
    poetry add PACKAGE
    ```

* Update dependencies

    ```
    poetry update (PACKAGE)
    ```

* Remove dependencies

    ```
    poetry remove PACKAGE
    ```