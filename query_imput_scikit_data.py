#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 10:45:24 2019

@author: valteresj
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# To use this experimental feature, we need to explicitly ask for it:
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.datasets import fetch_california_housing
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import datawig
import ImputMissing
from random import seed

N_SPLITS = 5

rng = np.random.RandomState(0)

X_full, y_full = fetch_california_housing(return_X_y=True)
# ~2k samples is enough for the purpose of the example.
# Remove the following two lines for a slower run with different error bars.
X_full = X_full[::10]
y_full = y_full[::10]
n_samples, n_features = X_full.shape

# Estimate the score on the entire dataset, with no missing values
br_estimator = BayesianRidge()
score_full_data = pd.DataFrame(
    cross_val_score(
        br_estimator, X_full, y_full, scoring='neg_mean_squared_error',
        cv=N_SPLITS
    ),
    columns=['Full Data']
)

# Add a single missing value to each row
X_missing = X_full.copy()
y_missing = y_full
missing_samples = np.arange(n_samples)
missing_features = rng.choice(n_features, n_samples, replace=True)
X_missing[missing_samples, missing_features] = np.nan

# Estimate the score after imputation (mean and median strategies)
score_simple_imputer = pd.DataFrame()
for strategy in ('mean', 'median'):
    estimator = make_pipeline(
        SimpleImputer(missing_values=np.nan, strategy=strategy),
        br_estimator
    )
    score_simple_imputer[strategy] = cross_val_score(
        estimator, X_missing, y_missing, scoring='neg_mean_squared_error',
        cv=N_SPLITS
    )

X_tech=pd.DataFrame(X_missing) 
X_tech.columns=['V_'+str(i) for i in X_tech.columns]
newdt=ImputMissing.IMV(data=X_tech)
newdata=newdt.ImputMissingValues()
estimators = [
    BayesianRidge(),
    DecisionTreeRegressor(max_features='sqrt', random_state=0),
    ExtraTreesRegressor(n_estimators=10, random_state=0),
    KNeighborsRegressor(n_neighbors=15)
]

score_TMI_imputer = pd.DataFrame()
for impute_estimator in estimators:
    score_TMI_imputer[impute_estimator.__class__.__name__] = \
        cross_val_score(
            impute_estimator, newdata, y_missing, scoring='neg_mean_squared_error',
            cv=N_SPLITS
        ) 

seed(7)
X_tech=pd.DataFrame(X_missing) 
X_tech.columns=['V_'+str(i) for i in X_tech.columns]
newdata1=X_tech.copy()
type_var=X_tech.dtypes
for i in list(X_tech.columns):
    if sum(pd.isna(X_tech[i])==True)>0 and type_var[i] in ['int64', 'float64']:
        imputer = datawig.SimpleImputer(
            input_columns=list(X_tech.drop(labels=i,axis=1).columns), # column(s) containing information about the column we want to impute
            output_column= i, # the column we'd like to impute values for
            output_path = 'imputer_model' # stores model data and metrics
            )
        imputer.fit(train_df=X_tech, num_epochs=50)
        
        imputed = imputer.predict(X_tech)
        newdata1.loc[np.where(pd.isna(X_tech[i])==True)[0],i]=imputed.iloc[np.where(pd.isna(X_tech[i])==True)[0],len(imputed.columns)-1]
    elif sum(pd.isna(X_tech[i])==True)>0 and type_var[i] not in ['int64', 'float64']:
        imputer = datawig.SimpleImputer(
            input_columns=list(X_tech.drop(labels=i,axis=1).columns), # column(s) containing information about the column we want to impute
            output_column= i, # the column we'd like to impute values for
            output_path = 'imputer_model' # stores model data and metrics
            )
        imputer.fit(train_df=X_tech, num_epochs=50)
        
        imputed = imputer.predict(X_tech)
        newdata1.loc[np.where(pd.isna(X_tech[i])==True)[0],i]=imputed.iloc[np.where(pd.isna(X_tech[i])==True)[0],len(imputed.columns)-2]
        
score_DataWig_imputer = pd.DataFrame()
for impute_estimator in estimators:
    score_DataWig_imputer[impute_estimator.__class__.__name__] = \
        cross_val_score(
            impute_estimator, newdata1, y_missing, scoring='neg_mean_squared_error',
            cv=N_SPLITS
        ) 




# Estimate the score after iterative imputation of the missing values
# with different estimators
estimators = [
    BayesianRidge(),
    DecisionTreeRegressor(max_features='sqrt', random_state=0),
    ExtraTreesRegressor(n_estimators=10, random_state=0),
    KNeighborsRegressor(n_neighbors=15)
]
score_iterative_imputer = pd.DataFrame()
for impute_estimator in estimators:
    estimator = make_pipeline(
        IterativeImputer(random_state=0, estimator=impute_estimator),
        br_estimator
    )
    score_iterative_imputer[impute_estimator.__class__.__name__] = \
        cross_val_score(
            estimator, X_missing, y_missing, scoring='neg_mean_squared_error',
            cv=N_SPLITS
        )
        


scores = pd.concat(
    [score_full_data, score_simple_imputer, score_iterative_imputer, pd.DataFrame(score_TMI_imputer), pd.DataFrame(score_DataWig_imputer)],
    keys=['Original', 'SimpleImputer', 'IterativeImputer', 'TMIImputer', 'DWImputer'], axis=1
)

# plot boston results
fig, ax = plt.subplots(figsize=(13, 6))
means = -scores.mean()
errors = scores.std()
means.plot.barh(xerr=errors, ax=ax)
ax.set_title('California Housing Regression with Different Imputation Methods')
ax.set_xlabel('MSE (smaller is better)')
ax.set_yticks(np.arange(means.shape[0]))
ax.set_yticklabels([" w/ ".join(label) for label in means.index.get_values()])
plt.tight_layout(pad=1)
plt.show()
