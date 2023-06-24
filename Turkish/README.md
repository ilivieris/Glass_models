# Controlled Vocabulary (CCV) recongition - Use case: GR

This is the implementation code for hyperparameter optimization framework for tuning sklearn models using Optuna for recognizing CCV classes. The dataset is splitted in a training set and a hold-out test. The tuning is performed on the training set (using k-fold stratified cross-validation) while the evaluation on the hold-out set.

Currently, the available model is *XGBoost* but the code can be easily modified to include any sklearn model. All the results, figures and models are logged in MLflow.

<br/>

## Optuna

**Optuna** is a hyperparameter optimization framework for tuning models. It lets you understand how hyperparameters affect your model and improves your model performance.

There are many samplers available to tune your models. It still contains the standard grid search and random search models. But, in addition, you can also choose :

- Tree-structured Parzen Estimator 
- A Quasi-Monte Carlo Sampler
- An Intersection Search Space Sampler

<br/>


## Data
---

The data concern a multi-class classification problem (CCV classes identification) is contained in ``Data``
and can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1RR-jJdNzljYKxXqHVVaWPtogKpHv-8D7)



<br/>

## Notebooks
---

- ``01. Data_preperation.ipynb``: parses the original data and transforms them in a suitable form for training a ML model. The embeddings are calculated using SpaCy library. Additionally, not all classes are considered in the classification problem since most of them can be implemented using hard-rules.
- ``02. Hard-rule evaluation.ipynb``: Evaluates the efficiency and accuracy of created hard-rules.
- ``03. XGBoost.ipynb`` hyperparameter tuning of XGBoost model information using LIME and SHAP methods.
- ``04. Model development.ipynb`` trains a XGBoost model with user selected hyper-paramater tuning on the whole dataset.
- ``05. Inference.ipynb``: code for providing inference on user's input.

<br/>

## How to run
--- 

1. Create a virtual environment 
```
    conda create -n myEnv python=3.8
```

2. Activate the virtual environment 
```
    conda activate myEnv
```
3. Install requirements 
```
    pip install -r requirements.txt
```
4. Unzip the corresponding zip files for obtaining the data

5. Run jupyter notebooks
```
    01. Data_preperation.ipynb
    02. XGBoost.ipynb
    03. Inference.ipynb
```


<br/>

## Requirements

- python==3.8
- lightgbm==3.3.5
- optuna==3.1.0
- numpy==1.23.5
- pandas==1.5.2
- scikit-learn==1.2.0
- xgboost==1.7.3
- lightgbm==3.3.5
- spacy==3.5.3
- turkspacy=0.1.0

<br/>

## :mailbox: Contact
---
Ioannis E. Livieris (livieris@novelcore.eu)
