<!-- markdownlint-disable -->
<h1 align="center">
   Practical Application Assignment 17.1: Comparing Classifiers 
    <br>
</h1>

<p align="center">
    <strong>🏆&nbsp; A comparison of classifiers: LogisticRegression, KNN, DecisionTree and Support Vector Machine.</strong>
</p>

<p align="center">
    <a href="https://github.com/pnanyaduba/kraftwerk/tree/main/practical_application_II_starter" title="Best-of-badge"><img src="https://ci.appveyor.com/api/projects/status/32r7s2skrgm9ubva?svg=true&passingText=master%20-%20OK"></a>
    <a href="#Contents" title="Project Count"><img src="https://img.shields.io/badge/projects-2nd-blue.svg?color=5ac4bf"></a>
    <a href="#Contribution" title="Contributions are welcome"><img src="https://img.shields.io/badge/contributions-welcome-green.svg"></a>
    <a href="#" title="Best-of Updates"><img src="https://img.shields.io/github/release-date/ml-tooling/best-of-ml-python?color=green&label=updated"></a>
    <a href="https://twitter.com/peteberc" title="Follow on Twitter"><img src="https://img.shields.io/twitter/follow/mltooling.svg?style=social&label=Follow"></a>
</p>

In this application, I explored a Bank Marketing Data Set from the following site: [UCI Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing). Our goal is to use about four classifiers in modelling the data and then compare the performance of each of the classifiers.

---

<p align="center">
     🧙‍♂️&nbsp; Click here to access the Project Notebook <a href="https://github.com/pnanyaduba/PortugueseBank/blob/main/PortugueseBank.ipynb">Practical App 3</a> <br>
    📫&nbsp; Subscribe to my Twitter handle <a href="http://twitter.com/hashtharg">@hashtharg</a> for updates and my tweets.
</p>

---


## Contents on Jupyter Notebook Steps

- [Jupyter Notebook Link](https://github.com/pnanyaduba/PortugueseBank/blob/main/PortugueseBank.ipynb)
- Information about the data
- Dropping Unwanted Columns
- Encode the whole categorical columns
- Get the Correlation Matrix
- Plot the Scatter Matrix
- Define the Modelling Data
- Perform Principal Component Analysis of the Scaled Data
- Create four models - Logistic Regression, DecisionTree, KNN and SVC
- Evaluate the four models
- Summarize the results of the Evaluations
- Select the Best Model out of the four based on the summary
- Create a Report of the Analysis

<br>

## Machine Learning Libraries Used on Jupyter Notebook

- statsmodels.tsa.filters.filtertools as ft
- sklearn.metrics import mean_squared_error
- statsmodels.tsa.filters.filtertools import convolution_filter
- sklearn.feature_selection import SequentialFeatureSelector
- statsmodels.tsa.seasonal import _extrapolate_trend
- pandas.testing as tm
- statsmodels.tsa.arima_process as arima_process
- statsmodels.graphics.tsaplots as tsaplots
- numpy as np
- pandas as pd
- matplotlib.pyplot as plt
- statsmodels.api as sm
- statsmodels.tsa.seasonal import seasonal_decompose
- sklearn.preprocessing import OneHotEncoder
- sklearn.pipeline import Pipeline
- sklearn.preprocessing import StandardScaler
- sklearn.impute import SimpleImputer
- sklearn.compose import ColumnTransformer
- sklearn.decomposition import PCA
- sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
- sklearn.pipeline import make_pipeline
- sklearn.model_selection import train_test_split, GridSearchCV
- sklearn.preprocessing import PolynomialFeatures, OrdinalEncoder
- sklearn.linear_model import Ridge
- scipy import stats
- scipy.linalg import svd
- warnings
- warnings.filterwarnings('ignore')
