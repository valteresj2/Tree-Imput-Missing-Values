# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 09:55:17 2019

@author: valter.e.junior
"""

import pandas as pd
import numpy as np
from info_gain import info_gain


def ab(x):
    return ''.join(x)

def aa(x):
    qtd_missing=sum(pd.isna(x))
    qtd_notmissing=sum(pd.isna(x)==False)
    media=np.nanmean(x)
    cols=['Mean','qtd_notmissing','qtd_missing']
    return pd.Series((media,qtd_notmissing,qtd_missing),index=cols)

def aa2(x,quantil):
    qtd_missing=sum(pd.isna(x))
    qtd_notmissing=sum(pd.isna(x)==False)
    if qtd_notmissing>0:
        x=[j for j in x if pd.isna(j)!=True]
        gg=pd.Series(pd.cut(x, quantil, include_lowest=True).astype(str))
        weigths=gg.value_counts()/len(x)
        var_weigths=pd.Series([0]*len(gg))
        for j in weigths.index:
            var_weigths[np.where(gg==j)[0]]=weigths[j]
        media=sum(np.array(x)*var_weigths)/sum(var_weigths)
    else:
        media=np.nanmean(x)
    cols=['Mean','qtd_notmissing','qtd_missing']
    return pd.Series((media,qtd_notmissing,qtd_missing),index=cols)

def bb(x,cols):
    qtd_missing=sum(pd.isna(x))
    qtd_notmissing=sum(pd.isna(x)==False)
    value=[]
    x=x.value_counts()
    for i,value1 in enumerate(cols):
        if sum(cols[i]==x.index)>0:
            value.append(x[np.where(x.index==cols[i])[0]].values[0])
        else:
            value.append(0)
    
    
    media=value
    lv=['qtd_notmissing','qtd_missing']
    cols=list(cols)
    cols.extend(lv)
    media.append(qtd_notmissing)
    media.append(qtd_missing)
    return pd.Series(media,index=cols)

class IMV(object):
    def __init__(self,data,target=np.nan,save_model=False):
        self.weights={}
        self.dt=data.copy()
        self.target=target
        self.pre_processing={}
        self.save_model=save_model
    def ImputMissingValues(self):
        
        type_var = self.dt.dtypes
        var_name=self.dt.columns
        if pd.isna(self.target)==False:
            type_var = type_var[np.where(self.dt.columns != self.target)[0]]
            var_name=self.dt.columns.drop(labels=self.target)
        
        dt_mod=self.dt.copy()
        
        for i, value in enumerate(var_name):
        #    if type_var[i] in ['int64', 'float64'] and sum(pd.isna(dt[value])) > 0:
        #        dt[value] = pd.fillna(dt[value].min())
            if type_var[value] not in ['int64', 'float64'] and sum(pd.isna(self.dt[value])) > 0:
                dt_mod.loc[np.where(pd.isna(self.dt[value])==True)[0],value] = 'MISSING'
                self.pre_processing[value]='MISSING'
            if type_var[value] in ['int64', 'float64']:
                newvalue=dt_mod[value]+0.01
                #quantil = np.unique(dt_mod[value].quantile(list(np.array(list(range(0, 120, 20))) / 100)))
                quantil = np.unique(newvalue.quantile(list(np.array(list(range(0, 125, 25))) / 100)))-0.01
                if len(quantil) > 2:
                    gg=pd.cut(dt_mod[value], quantil, include_lowest=True).astype(str)
                    if len(np.where(np.array(gg)==str(np.nan))[0])>0:
                        gg[np.where(np.array(gg)==str(np.nan))[0]]='MISSING'
                    dt_mod[value]=gg
                    self.pre_processing[value]=quantil
                else:
                    dt_mod[value] = dt_mod[value].astype(str)
                
                    
        valor=[]
        for i in var_name:
            valor.append(len(np.unique(dt_mod[i])))                    
#            if pd.isna(self.target)==False:
#                valor.append(info_gain.info_gain(list(dt_mod[self.target]), list(dt_mod[i])))
#            else:
#                valor.append(len(np.unique(dt_mod[i])))
            
        
        dt_eq=pd.DataFrame({'var':var_name,'value':valor})
        dt_eq=dt_eq.sort_values(by=['value'],ascending=True)
        var_name=dt_eq['var']
        for i in var_name:
            if type_var[i] in ['int64', 'float64'] and sum(dt_mod[i]=='MISSING')>0:
                  
                dt_mod['var_mod']=self.dt[i]
                var_name1=list(var_name.copy())
                var_name1.pop(np.where(var_name==i)[0][0])
                var_name1=pd.Series(var_name1)
                for index in range(1,len(var_name1)):
                #gg=dt_mod.groupby(list(var_name.drop(labels=i)), as_index=False)['var_mod'].apply(aa)
                    gg=dt_mod.groupby(list(var_name1[0:index]), as_index=False)['var_mod'].apply(aa)
                    #gg=dt_mod.groupby(list(var_name1[0:index]), as_index=False)['var_mod'].apply(aa2,self.pre_processing[i])
                    if sum(gg.iloc[np.where(pd.isna(gg['Mean'])==True)[0],2])/sum(pd.isna(self.dt[i]))>0.05:
                        
                        break
                gg.drop(labels=gg.index[list(np.where((gg['qtd_missing']==0) | (pd.isna(gg['Mean'])==True) | (gg.loc[:,'qtd_notmissing']<10))[0])],axis=0,inplace=True)
                conc_index=[''.join(k) for k in gg.index]
                ef=dt_mod.loc[np.where(pd.isna(self.dt[i]))[0],var_name1[0:index]].apply(ab,axis=1)
                index_ref_dt=[k1 for k1,value1 in enumerate(ef) for k2,value2 in enumerate(conc_index) if value1==value2]
                index_reff=[k2 for k1,value1 in enumerate(ef) for k2,value2 in enumerate(conc_index) if value1==value2]
                self.dt.loc[ef.index[index_ref_dt],i]=list(gg.loc[gg.index[index_reff],'Mean'])
                if sum(pd.isna(self.dt[i])==True)>0:
                    self.dt[i] = self.dt[i].fillna(self.dt[i].mean())
                self.weights[i]={'weights':gg,'features':var_name1[0:index]}
            
            if type_var[i] not in ['int64', 'float64'] and sum(dt_mod[i]=='MISSING')>0:
                
                dt_mod['var_mod']=self.dt[i]
                var_name1=list(var_name.copy())
                var_name1.pop(np.where(var_name==i)[0][0])
                var_name1=pd.Series(var_name1)
                for index in range(1,len(var_name1)):
                    
                    gg=dt_mod.groupby(list(var_name1[0:index]), as_index=False)['var_mod'].apply(bb,(self.dt[i].unique()))
                    if sum(gg.iloc[:,0:(len(gg.columns)-2)].apply(sum,axis=1)==0)>0 and sum(gg.loc[gg.index[np.where(gg.iloc[:,0:(len(gg.columns)-2)].apply(sum,axis=1)==0)[0]],'qtd_missing'])/sum(pd.isna(self.dt[i]))>=0.09:
                        if index>2:
                            gg=dt_mod.groupby(list(var_name1[0:(index-1)]), as_index=False)['var_mod'].apply(bb,(self.dt[i].unique()))
                            index=index-1
                        break
                
                gg.drop(labels=gg.index[list(np.where((gg['qtd_missing']==0) | (gg.iloc[:,0:(len(gg.columns)-2)].apply(sum,axis=1)==0) | (gg.loc[:,'qtd_notmissing']<30))[0])],axis=0,inplace=True)
                conc_index=[''.join(k) for k in gg.index]
                ef=dt_mod.loc[np.where(pd.isna(self.dt[i]))[0],var_name1[0:index]].apply(ab,axis=1)
                index_ref_dt=[k1 for k1,value1 in enumerate(ef) for k2,value2 in enumerate(conc_index) if value1==value2]
                index_reff=[k2 for k1,value1 in enumerate(ef) for k2,value2 in enumerate(conc_index) if value1==value2]  
                idxx=gg.iloc[:,0:(len(gg.columns)-2)].apply(np.argmax,axis=1).values
                self.dt.loc[ef.index[index_ref_dt],i]=list(idxx[index_reff])
                if sum(pd.isna(self.dt[i])==True)>0:
                    self.dt[i] = self.dt[i].fillna('others')
                self.weights[i]={'weights':gg,'features':var_name1[0:index]}
        
        if self.save_model==True:       
            np.save('Weights.npy', self.weights)
            np.save('Pre_processing.npy', self.pre_processing)
                    
        return self.dt
    

    def predict(self,new_dt=np.nan,new_weights=np.nan,new_pre_processing=np.nan):
        
        if pd.isna(new_weights)==True:
            new_weights=self.weights
            new_pre_processing=self.pre_processing
        type_var = new_dt.dtypes
        dt_mod=new_dt.copy()
        for i in dt_mod.columns:
            if sum(np.array(list(new_pre_processing.keys()))==i)>0:
                if len(np.where(new_dt.columns==i)[0])>0:
                    if type_var[i] in ['int64', 'float64']:
                        quantil=new_pre_processing[i]
                        #quantil[len(quantil)-1]=35
                        faixas=pd.cut(new_dt[i], quantil, include_lowest=True)#.astype(str)
                        if sum(faixas.isnull())>0:
                            minimo=np.nanmin(new_dt[i])
                            maximo=np.nanmax(new_dt[i])
                            faixas=faixas.astype(str)
                            if quantil[0]>minimo:
                                faixas[np.where(new_dt[i]<quantil[0])[0]]=str(np.unique(list(faixas.astype(str)))[0])
                            if quantil[len(quantil)-1]<maximo:
                                u_value=list(faixas.unique())
                                u_value=[k for k in u_value if str(quantil[len(quantil)-1]) in k][0]
                                faixas[np.where(new_dt[i]>quantil[len(quantil)-1])[0]]=u_value
                            faixas=pd.Series(faixas)
                        if sum(np.array(faixas)==str(np.nan))>0:
                            faixas[np.where(np.array(faixas)==str(np.nan))[0]] = 'MISSING'
                        dt_mod[i]=faixas.astype(str)
                    if type_var[i] not in ['int64', 'float64']:
                        if sum(pd.isna(dt_mod[i])==True)>0:
                            dt_mod[i] = dt_mod[i].fillna('MISSING')
                else:
                    dt_mod[i] = dt_mod[i].astype(str)
            else:
                dt_mod[i] = dt_mod[i].astype(str)
                     
                        
        for i in new_weights.keys():
            if type_var[i] in ['int64', 'float64'] and sum(dt_mod[i]=='MISSING')>0:
                gg=new_weights[i]['weights']
                var_name1=new_weights[i]['features']
                conc_index=[''.join(k) for k in gg.index]
                ef=dt_mod.loc[np.where(pd.isna(new_dt[i]))[0],var_name1].apply(ab,axis=1)
                index_ref_dt=[k1 for k1,value1 in enumerate(ef) for k2,value2 in enumerate(conc_index) if value1==value2]
                index_reff=[k2 for k1,value1 in enumerate(ef) for k2,value2 in enumerate(conc_index) if value1==value2]
                new_dt.loc[ef.index[index_ref_dt],i]=list(gg.loc[gg.index[index_reff],'Mean'])
                if sum(pd.isna(new_dt[i])==True)>0:
                    new_dt[i] = new_dt[i].fillna(new_dt[i].min())
            if type_var[i] not in ['int64', 'float64'] and sum(dt_mod[i]=='MISSING')>0:
                gg=new_weights[i]['weights']
                var_name1=new_weights[i]['features']
                gg.drop(labels=gg.index[list(np.where((gg['qtd_missing']==0) | (gg.iloc[:,0:(len(gg.columns)-2)].apply(sum,axis=1)==0))[0])],axis=0,inplace=True)
                conc_index=[''.join(k) for k in gg.index]
                ef=dt_mod.loc[np.where(pd.isna(new_dt[i]))[0],var_name1].apply(ab,axis=1)
                index_ref_dt=[k1 for k1,value1 in enumerate(ef) for k2,value2 in enumerate(conc_index) if value1==value2]
                index_reff=[k2 for k1,value1 in enumerate(ef) for k2,value2 in enumerate(conc_index) if value1==value2]  
                idxx=gg.iloc[:,0:(len(gg.columns)-2)].apply(np.argmax,axis=1).values
                new_dt.loc[ef.index[index_ref_dt],i]=list(idxx[index_reff])
                if sum(pd.isna(new_dt[i])==True)>0:
                    new_dt[i] = new_dt[i].fillna('others')
        return new_dt
    
    
        