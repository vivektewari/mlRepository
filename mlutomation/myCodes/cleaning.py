import pandas as pd
import numpy as np
from dataExploration import distReports

dataset='application_train'
df=pd.read_csv('/home/pooja/PycharmProjects/homeCredit/dataManagerFiles/train/'+dataset+'.csv')
if dataset=='application_train':

    df['AMT_INCOME_TOTAL']=df['AMT_INCOME_TOTAL'].apply(lambda row: 1000000 if row>1000000 else row)#cleaning
    df['DAYS_EMPLOYED']=df['DAYS_EMPLOYED'].apply(lambda row: np.nan if row==365243 else row)#cleaning

    df['CNT_FAM_MEMBERS']=df['CNT_FAM_MEMBERS'].fillna(0)
df.to_csv('/home/pooja/PycharmProjects/homeCredit/dataManagerFiles/train/cleaned/'+dataset+'.csv')
d=distReports(df)
d.to_csv('/home/pooja/PycharmProjects/homeCredit/dataManagerFiles/train/cleaned/describe'+dataset+'.csv')