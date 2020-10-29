from dataExploration import distReports,plotGrabh
import numpy as np
from quickRun import quick
from funcs import normalize,getCount,getMean
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

from numpy import mean
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from dataManager import dataOwner,dataObject
#from varManager import varOwner,varFactory
# from varclushi import VarClusHi
from iv import IV
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn import tree
from matplotlib import pyplot
from xgboost import XGBClassifier
from itertools import combinations
import warnings
method='NN'
case=1
pk='PassengerId'
baseLoc='/home/pooja/PycharmProjects/titanic/'

target='Survived'

pk='PassengerId'
if case==1:
    main=pd.read_csv(baseLoc+'baseDatasets/main.csv')
    test=pd.read_csv(baseLoc+'baseDatasets/test.csv')
    test[target]=np.nan
    training=main[pk]
    testing=test[pk]
    main=main.set_index(pk)
    validSample= main.sample(n=int(main.shape[0] * 0.00), random_state=1)
    #print(validSample.index)

    main=main.append(test.set_index(pk))
    main[target] = main.groupby(['Sex', 'Pclass'])[target].apply(lambda x: x.fillna(x.mean()))
    indexes=main[['Survived']]
    # missing treatments:
    main['Embarked'] = main['Embarked'].fillna('C')
    main['Age'] = main.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
    main['Fare'] = main.groupby(['Embarked', 'Pclass'])['Fare'].apply(lambda x: x.fillna(x.median()))


    #self feature engeneering
    main['Cabin'] = main['Cabin'].fillna('Z')
    str1 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    main['Cabin'] = main['Cabin'].map(str)
    main['cabinConvert'] = main.Cabin.apply(lambda x: x.strip(" ")[0].replace(x[0], str((str1.find(x[0]) + 1) * 1000)))
    initials = ['Dr.', ' Mr.', 'Master', 'Mrs.', 'Miss', 'Rev.', 'Don.', 'Major.', 'Lady.', 'Sir.', 'Col.', 'Mile.',
                'Capt.', 'Countess.']
    main['initials'] = main['Name'].apply(lambda row: ("").join([i for i in initials if str(row).find(i) != -1]))
    main['initials'] = main['initials'].apply(lambda row: row if row in [' Mr.', 'Master', 'Mrs.', 'Miss'] else 'pool')
    main['married']=main['initials'].apply(lambda row: 1 if row in [ 'Mrs.'] else 0 if row in ['Master','Miss'] else -1 if row in [" Mr."]  else -2)
    main['familySize']=main['Parch']+main['SibSp']
    main['familySize']=main['familySize'].apply(lambda row:0 if row==0 else 1 if row<2 else 2)
    #binning
    v=main['familySize'].value_counts()
    main = getMean(main, ['cabinConvert','Ticket'], target)
    main = getCount(main, ['Cabin', 'cabinConvert', 'Ticket','initials'])#,'initials'
    main['TicketMeanT']=main.apply(lambda row: 0 if row['TicketCount']<=1  else \
        1 if row['TicketMeanT']<0.5 or row['TicketMeanT'] in [0,1]  else 2,axis=1)
    te=main.loc[training].groupby(['TicketMeanT','Sex'])[target].agg(['mean','count','sum'])
    # main['cabinConvertMeanT'] = main.apply(lambda row: 1 if row['cabinConvertMeanT'] <0.50 else 2\
    #     if row['cabinConvertMeanT'] <0.50 else  3 if row['cabinConvertMeanT'] <1 else np.nan, axis=1)
    f=main[['cabinConvertMeanT',target]]
    main['cabinConvert']=main['cabinConvert'].map(int)
    main['Sex']=main['Sex'].apply(lambda x: 1 if x=='male' else 0)


    for feature in ['Age','Fare' ,'TicketCount','TicketMeanT']:
        main[feature] = pd.qcut(main[feature], 10,duplicates='drop')
        main[feature] = preprocessing.LabelEncoder().fit_transform(main[feature])
    main = pd.get_dummies(main,columns=['Embarked'])
    #plotGrabh(main.drop([ 'Name', 'Ticket', 'Cabin', 'initials'], axis=1) , target, baseLoc + "extraGraphs/")
    main = main.drop(['Survived', 'Name', 'Ticket', 'Cabin', 'initials'], axis=1)  # 'Embarked','Sex

    #main['Survived'] = indexes['Survived']

    testSample=main.loc[testing]
    main=main.loc[training]
    main.to_csv(baseLoc+'dataManagerFiles/train/'+"mainWithDummies.csv")
    testSample.to_csv(baseLoc+'dataManagerFiles/train/'+"testWithDummies.csv")

    quick()
case=2
if case==2:
    main=pd.read_csv(baseLoc+'dataManagerFiles/train/'+"mainwithCovars.csv",index_col=pk)
    tar= pd.read_csv(baseLoc + 'dataManagerFiles/train/' + "target.csv", index_col=pk)
    main=main.join(tar)
    testSample = pd.read_csv(baseLoc + 'dataManagerFiles/train/' + "testwithCovars.csv",index_col=pk)
    testSample.columns=[name.replace("testWithDummies","mainWithDummies") for name in testSample.columns]
if case==2:
    test=main.sample(n=int(main.shape[0]*0.20) ,random_state=1)
    #print(test.index)
    train=main.drop(test.index,axis=0)
    test_y=test[target]
    train_y=train[target]
    exclu=['initialsCount','0cabin','0Fare|Sex','0Sex|Fare','0Sex|cabin','0initialsCount','0Sex|Cabin','0cabinConvertMeanT|SexdivmainWithDummies','0TicketMean']

    extraExcludes=[v for v in train.columns if any([str(v).find(c)>=0 for c in exclu  ])]
    train=train.drop([target]+extraExcludes,axis=1)
    test=test.drop([target]+extraExcludes,axis=1)

    X=train
    y=train_y

    model = XGBClassifier()
        #model=RandomForestClassifier(n_estimators=50, max_features='sqrt')
    #varSelected=['SibSp0mainWithDummies', 'Fare|PclassdivmainWithDummies', 'Fare|cabinConvertCountdivmainWithDummies', 'Age|CabinCountdivmainWithDummies', 'Age|Embarked_SdivmainWithDummies', 'Sex|TicketCountdivmainWithDummies', 'cabinConvert|TicketCountdivmainWithDummies', 'Pclass|cabinConvertMeanTdivmainWithDummies', 'cabinConvertMeanT|SexdivmainWithDummies', 'Fare|TicketCountmultmainWithDummies', 'Fare|Embarked_CmultmainWithDummies', 'Age|TicketCountmultmainWithDummies', 'Age|PclassmultmainWithDummies', 'Parch|PclassmultmainWithDummies', 'CabinCount|PclassmultmainWithDummies', 'Pclass|Embarked_SmultmainWithDummies']
    varSelected=['Age0mainWithDummies', 'SibSp0mainWithDummies', 'cabinConvertCount0mainWithDummies', 'Age|SibSpdivmainWithDummies', 'Age|Embarked_CdivmainWithDummies', 'Age|Embarked_SdivmainWithDummies', 'Age|ParchdivmainWithDummies', 'Age|cabinConvertdivmainWithDummies', 'Age|TicketCountdivmainWithDummies', 'Age|cabinConvertCountdivmainWithDummies', 'Age|CabinCountdivmainWithDummies', 'Fare|SibSpdivmainWithDummies', 'Fare|cabinConvertMeanTdivmainWithDummies', 'Fare|PclassdivmainWithDummies', 'Fare|Embarked_CdivmainWithDummies', 'Fare|Embarked_SdivmainWithDummies', 'Fare|cabinConvertdivmainWithDummies', 'Fare|TicketCountdivmainWithDummies', 'Fare|cabinConvertCountdivmainWithDummies', 'Fare|CabinCountdivmainWithDummies', 'Sex|cabinConvertMeanTdivmainWithDummies', 'Sex|TicketCountdivmainWithDummies', 'SibSp|Embarked_CdivmainWithDummies', 'SibSp|ParchdivmainWithDummies', 'cabinConvertMeanT|PclassdivmainWithDummies', 'cabinConvertMeanT|TicketCountdivmainWithDummies', 'Pclass|cabinConvertdivmainWithDummies', 'Embarked_C|ParchdivmainWithDummies', 'cabinConvert|TicketCountdivmainWithDummies', 'Age|FaremultmainWithDummies', 'Age|SexmultmainWithDummies', 'Age|cabinConvertMeanTmultmainWithDummies', 'Age|PclassmultmainWithDummies', 'Age|Embarked_SmultmainWithDummies', 'Age|Embarked_QmultmainWithDummies', 'Age|cabinConvertmultmainWithDummies', 'Age|TicketCountmultmainWithDummies', 'Age|cabinConvertCountmultmainWithDummies', 'Fare|Embarked_CmultmainWithDummies', 'Fare|Embarked_SmultmainWithDummies', 'Fare|ParchmultmainWithDummies', 'Fare|TicketCountmultmainWithDummies', 'SibSp|CabinCountmultmainWithDummies', 'Pclass|Embarked_SmultmainWithDummies', 'Pclass|ParchmultmainWithDummies', 'Pclass|CabinCountmultmainWithDummies', 'Embarked_S|ParchmultmainWithDummies', 'Embarked_S|TicketCountmultmainWithDummies']

    varSelected=X.columns
    # for n in range(10,30):
    #     combs = list(combinations(list(X.columns), n))
    #     varComb=list(combs)
    #     for varSelected in varComb:
    t=X[varSelected]
    t['actual']=y
    model.fit(X[varSelected], y, eval_metric="logloss",eval_set=[(test[varSelected],test_y)],\
              early_stopping_rounds=30)
    # model = RandomForestClassifier(n_estimators=100, max_depth=8, max_features='sqrt', verbose=1, bootstrap=False)
    # X=X.replace([np.nan,np.inf,-np.inf],-579)
    # test =test.replace([np.nan, np.inf, -np.inf], -579)
    # testSample = testSample.replace([np.nan, np.inf, -np.inf], -579)
   # model.fit(X[varSelected], y)#eval_metric="logloss",, eval_set=[(test[varSelected], test_y)],early_stopping_rounds=30 \

    # cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1)
    # #evaluate model
    # scores = cross_val_score(model, X[varSelected], y, scoring='roc_auc', cv=cv, n_jobs=-1)
    # print('Mean ROC AUC: %.5f' % mean(scores))

    # for col in testSample.columns:
    #     if col not in main.columns:
    #         print(col)
    #print(main.columns)
    #print(main.columns)
    a=model.predict(testSample[varSelected])
    predicted=testSample[[]]

    tas=accuracy_score(y,model.predict(X[varSelected]))
    vas=accuracy_score(test_y,model.predict(test[varSelected]))
    print(tas, vas)
    if False:
        test['predicted']=model.predict(test[varSelected])
        test['actual']=test_y
        test['diff']=test['predicted']-test['actual']
        e= test[test['diff']!=0]
        ne=test[test['diff']==0]
        distReports(e).to_csv(baseLoc+"errors.csv")
        distReports(ne).to_csv(baseLoc + "noterrors.csv")
        distReports(testSample).to_csv(baseLoc+"distTest.csv")


    predicted[target]=a
    predicted.index.name=pk
    predicted=predicted[[target]]
    predicted.to_csv(baseLoc+"predicted.csv")

    if True:
        importance = model.feature_importances_
        # summarize feature importance
        goodVar=[]
        goodImp=[]
        var=X[varSelected].columns
        for i, v in enumerate(importance):
            #print(varSelected[i]+', Score: %.5f' % ( v))
            if v>0.005:
                print(v,var[i])
                goodVar.append(var[i])
                goodImp.append(v)
        print(len(goodVar),len(var))
        print(goodVar)
        goodVar=[name.replace('mainWithDummies','') for name in goodVar]
        # plot feature importance
        #pyplot.bar([x for x in range(len(importance))], importance)
        #pyplot.barh(goodVar, goodImp)

        pyplot.show()
