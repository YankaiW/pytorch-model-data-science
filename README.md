# Data Science with PyTorch

## Contents

- [Data Science with PyTorch](#data-science-with-pytorch)
  - [Contents](#contents)
  - [Introduction](#introduction)
  - [Repo Structure](#repo-structure)
    - [`pytorch_model_data_science`](#pytorch_model_data_science)
    - [`notebook`](#notebook)
  - [PyTorch Hyperparameter Tuning](#pytorch-hyperparameter-tuning)
  - [Development](#development)

## Introduction

With the release of PyTorch 2.0, more and more DL models are built based on PyTorch. Because of its flexibilty during model construction and its light-weight in package size, it beccomes a more and more popular tool in DL study and production.

This repo is a code-based collection, including, 

* PyTorch model construction.
* PyTorch model training and validation.
* PyTorch model hyperparameter tuning.

which can help beginers use PyTorch models in study and production.

## Repo Structure

```
.
├── notebook
├── pytorch_model_data_science
└── tests
```

### `pytorch_model_data_science`

```
pytorch_model_data_science
├── model.py
├── train.py
└── validation.py
```

* `model.py`: The module storing the models for Deep Learning and the models 
  used for model packaging and tuning, including,
    * `DNN1DNet`: The model used for regression and classification based on DNN
      structure.
    * `CNN1DNet`: The model used for regression and classification based on CNN
      structure.
    * `EarlyStopper`: The class used for early-stopping when hyperparameter 
      tuning.
    * `PyTorchEstimator`: The class used to package PyTorch model as the 
      scikit-learn model, which can be deployed on the cloud-based platforms.
* `train.py`: The functions used for traning regression and classification 
  models.
* `validation.py`: The functions used to calculate metrics for regression and
  classification models.

### `notebook`

```
notebook
├── classifier_fashionmnist_1D.ipynb
├── regressor_california_housing_1D.ipynb
└── utils
    ├── model_pipe.ipynb
    └── utils.ipynb
```

* `utils/model_pipe.ipynb`: The DL models and the function used to train models.
* `utils/utils.ipynb`: The classes used for hyperparameter tuning.
* Examples:
  * `classifier_fashionmnist_1D.ipynb`: The example to use `FashionMNIST` 
    dataset to build 1D DNN and CNN classification models.
  * `regressor_california_housing_1D.ipynb`: The example to use 
    `California_Housing` dataset to build 1D DNN and CNN regression models.

## PyTorch Hyperparameter Tuning

When implementing hyperparameter tuning, one can refer to the examples that give
us the detailed steps of tuning for classification and regression problems, 

* `notebook/classifier_fashionmnist_1D.ipynb`
* `regressor_california_housing_1D.ipynb`

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