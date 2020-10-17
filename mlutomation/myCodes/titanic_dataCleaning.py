from dataExploration import distReports,plotGrabh
from dataManager import dataOwner,dataObject
#from varManager import varOwner,varFactory
# from varclushi import VarClusHi
from iv import IV
import pandas as pd
import warnings


baseLoc='/home/pooja/PycharmProjects/titanic/'
stage=9 #1
target='Survived'
pk='PassengerId'
main=pd.read_csv(baseLoc+'baseDatasets/main.csv')
indexes=main[['Survived','PassengerId']]
main=main.drop(['Survived','PassengerId','Name','Ticket','Cabin'],axis=1)
main=pd.get_dummies(main)
main['target']=main['Survived']
main['PassengerId']=main['PassengerId']

b=0