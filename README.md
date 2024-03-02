# Data Science with PyTorch

## Contents

- [Data Science with PyTorch](#data-science-with-pytorch)
  - [Contents](#contents)
  - [Introduction](#introduction)
  - [Usage](#usage)
    - [PyTroch Model Construction](#pytroch-model-construction)
    - [PyTorch Model Training and Validation](#pytorch-model-training-and-validation)
    - [PyTorch Model Hyperparameter Tuning](#pytorch-model-hyperparameter-tuning)
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
  * CNNNetRegressor
* Classifier
  * DNN1DClassifier
  * CNN1DClassifier

which can be invoked from there directly when needed.

### PyTorch Model Training and Validation

In the modules `pytorch_model_data_science.train` and `pytorch_model_data_science.validation`, there are functions used for model training and validation.

### PyTorch Model Hyperparameter Tuning

When implementing hyperparameter tuning, one can refer to the examples that give
us the detailed steps of tuning for classification and regression problems, 

* `notebook/classifier_fashionmnist_1D.ipynb`: The example to implement 
  hyperparameter tuning for 1D classification problem based on FashionMNIST 
  dataset.

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
    poetry update PACKAGE
    ```

* Remove dependencies

    ```
    poetry remove PACKAGE
    ```