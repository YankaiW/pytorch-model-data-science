# pytorch-model-data-science

## Contents

- [pytorch-model-data-science](#pytorch-model-data-science)
  - [Contents](#contents)
  - [Introduction](#introduction)
  - [Usage](#usage)
    - [PyTroch Model Construction](#pytroch-model-construction)
    - [PyTorch Model Training and Testing](#pytorch-model-training-and-testing)
    - [PyTorch Model Hyperparameter Tuning](#pytorch-model-hyperparameter-tuning)
  - [Development](#development)

## Introduction

With the release of PyTorch 2.0, more and more DL models are built based on PyTorch. Because of its flexibilty during model construction and its light weight in package size, it beccomes a more and more popular tool in DL study and production.

This repo is a code-based collection, including, 

* PyTorch model construction.
* PyTorch model training and testing.
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

### PyTorch Model Training and Testing

In the modules `pytorch_model_data_science.train` and `pytorch_model_data_science.validation`, there are functions used for model training and validation.

### PyTorch Model Hyperparameter Tuning

In the modules `pytorch_model_data_science.tune`, there is trainable function `tune_classifier`, which can be used in hyperparameter tuning for a classification model.

## Development

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