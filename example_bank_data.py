# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 16:08:49 2019

@author: valter.e.junior
"""

import numpy as np
import pandas as pd
import ImputMissing2
import ImputMissing
import os
import glob
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from impyute.imputation.cs import mice
import datawig
import random
from info_gain import info_gain


ROOT_DIR = os.path.abspath(os.curdir)

files=glob.glob(ROOT_DIR+'/Data/*')
files=np.sort(files)

alvo=[14,19,'y',8]
kk=files[2]
index=2
data = pd.read_csv(kk,sep=';')#,header=-1)
#data.replace('?',None,inplace=True)

#data[[1,2,3,4,5]] = data[[1,2,3,4,5]].replace(0, np.nan)
#data.columns=['V_'+str(i) for i in range(len(data.columns))]

#target='V_'+str(alvo[index])
#idx=random.sample(list(range(len(data))),round(0.2*len(data)))
#data.loc[idx,'contact']=None
data.loc[np.where(data['poutcome']=='unknown')[0],'poutcome']=None
data.loc[np.where(data['job']=='unknown')[0],'job']=None
data.loc[np.where(data['education']=='unknown')[0],'education']=None
data.loc[np.where(data['contact']=='unknown')[0],'contact']=None
data.loc[np.where(data['pdays']==-1)[0],'pdays']=None
target='y'

data[target]=np.where(data[target]==np.unique(data[target])[0],1,0)

newdt=ImputMissing.IMV(data=data,target=target)
newdata=newdt.ImputMissingValues()


#imputed_training=mice(pd.get_dummies(data.drop(labels=target,axis=1)).values)

newdata1=data.copy()
type_var=data.dtypes
for i in list(data.drop(labels=target,axis=1).columns):
    if sum(pd.isna(data[i])==True)>0 and type_var[i] in ['int64', 'float64']:
        imputer = datawig.SimpleImputer(
            input_columns=list(data.drop(labels=target,axis=1).columns), # column(s) containing information about the column we want to impute
            output_column= i, # the column we'd like to impute values for
            output_path = 'imputer_model' # stores model data and metrics
            )
        imputer.fit(train_df=data, num_epochs=50)
        
        imputed = imputer.predict(data)
        newdata1.loc[np.where(pd.isna(data[i])==True)[0],i]=imputed.iloc[np.where(pd.isna(data[i])==True)[0],len(imputed.columns)-1]
    elif sum(pd.isna(data[i])==True)>0 and type_var[i] not in ['int64', 'float64']:
        imputer = datawig.SimpleImputer(
            input_columns=list(data.drop(labels=target,axis=1).columns), # column(s) containing information about the column we want to impute
            output_column= i, # the column we'd like to impute values for
            output_path = 'imputer_model' # stores model data and metrics
            )
        imputer.fit(train_df=data, num_epochs=50)
        
        imputed = imputer.predict(data)
        newdata1.loc[np.where(pd.isna(data[i])==True)[0],i]=imputed.iloc[np.where(pd.isna(data[i])==True)[0],len(imputed.columns)-2]
        
        
orig=[]
tech=[]
deep=[]
missing=[]
var=[]
for i in data.columns:
    if type_var[i] not in ['int64', 'float64'] and sum(pd.isna(data[i])==True)>0:
        var.append(i)
        orig.append(info_gain.info_gain(list(data[target]), list(data[i])))
        tech.append(info_gain.info_gain(list(newdata[target]), list(newdata[i])))
        missing.append(sum(pd.isna(data[i])==True)/len(data))
        #deep.append(info_gain.info_gain(list(data[target]), list(newdata1[i])))
    

ixx=np.where(pd.isna(data[i])==True)[0]      
newdata.loc[ixx,i]
newdata1.loc[ixx,i] 
result=pd.DataFrame({'size_missing':missing,'Var':var,'Orig':orig,'Tech':tech})#,'Deep':deep})



newdata2=data.copy()
for i in data.columns:
    if sum(pd.isna(newdata2[i])==True)>0 and type_var[i]in ['int64', 'float64']:
        newdata2[i]=newdata2[i].fillna(newdata2[i].min())
    elif sum(pd.isna(newdata2[i])==True)>0 and type_var[i] not in ['int64', 'float64']:
        newdata2[i]=newdata2[i].fillna('others')

values = pd.get_dummies(newdata2.drop(labels=target,axis=1)).values
X = values
y = newdata2[target].values
# evaluate an LDA model on the dataset using k-fold cross validation
model = LinearDiscriminantAnalysis()
kfold = KFold(n_splits=10, random_state=7)
result = cross_val_score(model, X, y, cv=kfold, scoring='roc_auc')
print(result.mean(),' Result min')
print(result.std(),' Result min')


values = pd.get_dummies(newdata.drop(labels=target,axis=1)).values
X = values
y = newdata[target].values
# evaluate an LDA model on the dataset using k-fold cross validation
model = LinearDiscriminantAnalysis()
kfold = KFold(n_splits=10, random_state=7)
result = cross_val_score(model, X, y, cv=kfold, scoring='roc_auc')
print(result.mean(),' Result tech')
print(result.std(),' Result tech')


values = pd.get_dummies(newdata1.drop(labels=target,axis=1)).values
X = values
y = newdata[target].values
# evaluate an LDA model on the dataset using k-fold cross validation
model = LinearDiscriminantAnalysis()
kfold = KFold(n_splits=10, random_state=7)
result = cross_val_score(model, X, y, cv=kfold, scoring='roc_auc')
print(result.mean(),' Result deep')
print(result.std(),' Result deep')