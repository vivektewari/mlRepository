import pandas as pd
import numpy as np
from dataExploration import distReports,catGrouper
import os
dataset='remove Unamed 0'
folder="train/"
if dataset=='remove Unamed 0':
    files = os.listdir('/home/pooja/PycharmProjects/homeCredit/dataManagerFiles/'+folder+'cleaned/')
    for file in files:
        if ".csv" in file:
            df=pd.read_csv('/home/pooja/PycharmProjects/homeCredit/dataManagerFiles/'+folder+'cleaned/'+file)
            if 'Unnamed: 0' in df.columns:
                df=df.drop('Unnamed: 0',axis=1)
                df.to_csv('/home/pooja/PycharmProjects/homeCredit/dataManagerFiles/'+folder+'cleaned/'+file,index=False)
#
# df=pd.read_csv('/home/pooja/PycharmProjects/homeCredit/dataManagerFiles/'+folder+dataset+'.csv')
# if dataset=='application_train':
#
#     df['AMT_INCOME_TOTAL']=df['AMT_INCOME_TOTAL'].apply(lambda row: 1000000 if row>1000000 else row)#cleaning
#     df['DAYS_EMPLOYED']=df['DAYS_EMPLOYED'].apply(lambda row: np.nan if row==365243 else row)#cleaning
#
#     df['CNT_FAM_MEMBERS']=df['CNT_FAM_MEMBERS'].fillna(0)
#
# elif dataset=='previous_application':
#     d = df
#     varDict = {'NAME_CASH_LOAN_PURPOSE': ['XAP', 'XNA'], 'CODE_REJECT_REASON': ['XAP', ['HC', 'LIMIT', 'SCOFR']],
#                'NAME_TYPE_SUITE': [['Family', 'Children']], 'NAME_GOODS_CATEGORY': ['XNA'],
#                'NAME_GOODS_CATEGORY': [['Mobile', 'Computers', 'XNA']], 'NAME_PAYMENT_TYPE': ['Cash through the bank'],
#                'CODE_REJECT_REASON': ['XAP'], 'NAME_PORTFOLIO': ['POS'],
#                'NAME_PRODUCT_TYPE': ['walk-in'],
#                'CHANNEL_TYPE': ['Credit and cash offices', 'Contact center', 'AP+ (Cash loan)'],
#                'NAME_SELLER_INDUSTRY': ['XNA', 'Connectivity'],
#                'NAME_YIELD_GROUP': [['low_action', 'low_normal'], ['high', 'XNA']], 'PRODUCT_COMBINATION': [['POS industry with interest', 'POS household with interest', 'POS other with interest',
#              'POS mobile with interest', 'Cash Street: middle', 'Cash X-Sell: high', 'Cash Street: high', 'Card Street','Card X-Sell', 'Cash']],'FLAG_LAST_APPL_PER_CONTRACT':['Y'],'NFLAG_INSURED_ON_APPROVAL':[1, 0],
#                'NAME_CONTRACT_STATUS':[['Canceled','Refused']],'NAME_CONTRACT_TYPE':['Consumer loans'],'NAME_CLIENT_TYPE':['Refreshed']}
#     df = catGrouper(d, varDict)
#     df=df.drop(['WEEKDAY_APPR_PROCESS_START','HOUR_APPR_PROCESS_START','NFLAG_LAST_APPL_IN_DAY','RATE_INTEREST_PRIMARY','RATE_INTEREST_PRIVILEGED'],axis=1)
# elif dataset=='credit_card_balance':
#     d = df
#     varDict = {'NAME_CONTRACT_STATUS': ['Active']}
#     df = catGrouper(d, varDict)
# elif dataset=='POS_CASH_balance':
#     d = df
#     varDict = {'NAME_CONTRACT_STATUS': ['Active']}
#     df = catGrouper(d, varDict)
#
# elif dataset=='bureau':
#     pass
#     d = df
#     varDict = {'CREDIT_ACTIVE': ['Active'],'CREDIT_TYPE':['Consumer credit','Credit card']}
#     df = catGrouper(d, varDict)
#     df=df.drop(['CREDIT_CURRENCY'],axis=1)
# elif dataset=='bureau_balance':
#     d = df
#     d['STATUS_X'] = d['STATUS'].apply(lambda x: 1 if x == 'X' else 0)
#     d['STATUS_C'] = d['STATUS'].apply(lambda x: 1 if x == 'C' else 0)
#     d['STATUS']=d['STATUS'].replace(['X','C'],0).map(int)
#
#     df=d
#     # varDict = {'NAME_CONTRACT_STATUS': ['Completed']}
#     # df = catGrouper(d, varDict)
# elif dataset=='installments_payments':
#     pass
#     # d = df
#     # varDict = {'NAME_CONTRACT_STATUS': ['Completed']}
#     # df = catGrouper(d, varDict)
# df.to_csv('/home/pooja/PycharmProjects/homeCredit/dataManagerFiles/'+folder+'cleaned/'+dataset+'.csv')
# d=distReports(df,detail=True)
# d.to_csv('/home/pooja/PycharmProjects/homeCredit/dataManagerFiles/'+folder+'cleaned/'+dataset+'Describe.csv')
#
