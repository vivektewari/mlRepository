from dataExploration import distReports,plotGrabh
import numpy as np
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
from quickRun import quick
from funcs import normalize,getCount,getMean
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
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
pk='PassengerId'
baseLoc='/home/pooja/PycharmProjects/titanic/'

target='Survived'

pk='PassengerId'





main=pd.read_csv(baseLoc+'dataManagerFiles/train/'+"mainwithCovars.csv",index_col=pk)
tar= pd.read_csv(baseLoc + 'dataManagerFiles/train/' + "target.csv", index_col=pk)
main=main.join(tar)
testSample = pd.read_csv(baseLoc + 'dataManagerFiles/train/' + "testwithCovars.csv",index_col=pk)
testSample.columns=[name.replace("testWithDummies","mainWithDummies") for name in testSample.columns]

test=main.sample(n=int(main.shape[0]*0.20) ,random_state=1)
#print(test.index)
train=main.drop(test.index,axis=0)
test_y=test[target]
train_y=train[[target]]
exclu=['0Sex','0Ticket','0cabin','0Fare|Sex','0Sex|Fare','0Sex|cabin','initialsCount','0Sex|Cabin','0cabinConvertMeanT|SexdivmainWithDummies','0TicketMean']

extraExcludes=[v for v in train.columns if any([str(v).find(c)>=0 for c in exclu  ])]
train=train.drop([target]+extraExcludes,axis=1)
test=test.drop([target]+extraExcludes,axis=1)
testSample=testSample.drop(extraExcludes,axis=1)
X=train
y=train_y

leaderboard_model = RandomForestClassifier(criterion='gini',
                                           n_estimators=1100,
                                           max_depth=5,
                                           min_samples_split=4,
                                           min_samples_leaf=5,
                                           max_features='auto',
                                           oob_score=True,
                                           random_state=1,
                                           n_jobs=-1,
                                           verbose=1)
#model=RandomForestClassifier(n_estimators=50, max_features='sqrt')
#varSelected=['SibSp0mainWithDummies', 'Fare|PclassdivmainWithDummies', 'Fare|cabinConvertCountdivmainWithDummies', 'Age|CabinCountdivmainWithDummies', 'Age|Embarked_SdivmainWithDummies', 'Sex|TicketCountdivmainWithDummies', 'cabinConvert|TicketCountdivmainWithDummies', 'Pclass|cabinConvertMeanTdivmainWithDummies', 'cabinConvertMeanT|SexdivmainWithDummies', 'Fare|TicketCountmultmainWithDummies', 'Fare|Embarked_CmultmainWithDummies', 'Age|TicketCountmultmainWithDummies', 'Age|PclassmultmainWithDummies', 'Parch|PclassmultmainWithDummies', 'CabinCount|PclassmultmainWithDummies', 'Pclass|Embarked_SmultmainWithDummies']
varSelected=['Age0mainWithDummies', 'SibSp0mainWithDummies', 'cabinConvertCount0mainWithDummies', 'Age|SibSpdivmainWithDummies', 'Age|Embarked_CdivmainWithDummies', 'Age|Embarked_SdivmainWithDummies', 'Age|ParchdivmainWithDummies', 'Age|cabinConvertdivmainWithDummies', 'Age|TicketCountdivmainWithDummies', 'Age|cabinConvertCountdivmainWithDummies', 'Age|CabinCountdivmainWithDummies', 'Fare|SibSpdivmainWithDummies', 'Fare|cabinConvertMeanTdivmainWithDummies', 'Fare|PclassdivmainWithDummies', 'Fare|Embarked_CdivmainWithDummies', 'Fare|Embarked_SdivmainWithDummies', 'Fare|cabinConvertdivmainWithDummies', 'Fare|TicketCountdivmainWithDummies', 'Fare|cabinConvertCountdivmainWithDummies', 'Fare|CabinCountdivmainWithDummies', 'Sex|cabinConvertMeanTdivmainWithDummies', 'Sex|TicketCountdivmainWithDummies', 'SibSp|Embarked_CdivmainWithDummies', 'SibSp|ParchdivmainWithDummies', 'cabinConvertMeanT|PclassdivmainWithDummies', 'cabinConvertMeanT|TicketCountdivmainWithDummies', 'Pclass|cabinConvertdivmainWithDummies', 'Embarked_C|ParchdivmainWithDummies', 'cabinConvert|TicketCountdivmainWithDummies', 'Age|FaremultmainWithDummies', 'Age|SexmultmainWithDummies', 'Age|cabinConvertMeanTmultmainWithDummies', 'Age|PclassmultmainWithDummies', 'Age|Embarked_SmultmainWithDummies', 'Age|Embarked_QmultmainWithDummies', 'Age|cabinConvertmultmainWithDummies', 'Age|TicketCountmultmainWithDummies', 'Age|cabinConvertCountmultmainWithDummies', 'Fare|Embarked_CmultmainWithDummies', 'Fare|Embarked_SmultmainWithDummies', 'Fare|ParchmultmainWithDummies', 'Fare|TicketCountmultmainWithDummies', 'SibSp|CabinCountmultmainWithDummies', 'Pclass|Embarked_SmultmainWithDummies', 'Pclass|ParchmultmainWithDummies', 'Pclass|CabinCountmultmainWithDummies', 'Embarked_S|ParchmultmainWithDummies', 'Embarked_S|TicketCountmultmainWithDummies']

varSelected=X.columns

# for n in range(10,30):
#     combs = list(combinations(list(X.columns), n))
#     varComb=list(combs)
#     for varSelected in varComb:
N = 2
oob = 0
a=preprocessing.StandardScaler()
X_train= pd.DataFrame(a.fit_transform(X),index=X.index ,columns=X.columns)
X_test=pd.DataFrame(a.transform(test),index=test.index ,columns=test.columns)
df_all=train
y_train=y

probs = pd.DataFrame(np.zeros((len(X_test), N * 2)),
                     columns=['Fold_{}_Prob_{}'.format(i, j) for i in range(1, N + 1) for j in range(2)])
importances = pd.DataFrame(np.zeros((X_train.shape[1], N)), columns=['Fold_{}'.format(i) for i in range(1, N + 1)],
                           index=df_all.columns)
fprs, tprs, scores = [], [], []

skf = StratifiedKFold(n_splits=N, random_state=N, shuffle=True)

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    print('Fold {}\n'.format(fold))

    # Fitting the model
    leaderboard_model.fit(X_train.iloc[trn_idx], y_train.iloc[trn_idx])

    # Computing Train AUC score
    trn_fpr, trn_tpr, trn_thresholds = roc_curve(y_train.iloc[trn_idx],
                                                 leaderboard_model.predict_proba(X_train.iloc[trn_idx])[:, 1])
    trn_auc_score = auc(trn_fpr, trn_tpr)
    # Computing Validation AUC score
    val_fpr, val_tpr, val_thresholds = roc_curve(y_train.iloc[val_idx],
                                                 leaderboard_model.predict_proba(X_train.iloc[val_idx])[:, 1])
    val_auc_score = auc(val_fpr, val_tpr)

    scores.append((trn_auc_score, val_auc_score))
    fprs.append(val_fpr)
    tprs.append(val_tpr)

    # X_test probabilities
    probs.loc[:, 'Fold_{}_Prob_0'.format(fold)] = leaderboard_model.predict_proba(X_test)[:, 0]
    probs.loc[:, 'Fold_{}_Prob_1'.format(fold)] = leaderboard_model.predict_proba(X_test)[:, 1]
    importances.iloc[:, fold - 1] = leaderboard_model.feature_importances_

    oob += leaderboard_model.oob_score_ / N
    print('Fold {} OOB Score: {}\n'.format(fold, leaderboard_model.oob_score_))


importances['Mean_Importance'] = importances.mean(axis=1)
importances.sort_values(by='Mean_Importance', inplace=True, ascending=False)

pyplot.figure(figsize=(10, 10))
sns.barplot(x='Mean_Importance', y=importances.index, data=importances)

pyplot.xlabel('')
pyplot.tick_params(axis='x', labelsize=5)
pyplot.tick_params(axis='y', labelsize=5)
pyplot.title('Random Forest Classifier Mean Feature Importance Between Folds', size=5)

pyplot.show()


print('Average OOB Score: {}'.format(oob))
model=leaderboard_model

tas=accuracy_score(y,model.predict(a.transform(X)))
vas=accuracy_score(test_y,model.predict(a.transform(test)))
print(tas, vas)

if True:
    test['predicted']=model.predict(a.transform(test))
    test['actual']=test_y
    test['diff']=test['predicted']-test['actual']
    e= test[test['diff']!=0]
    ne=test[test['diff']==0]
    e.to_csv(baseLoc + "errorsObs.csv")
    distReports(e).to_csv(baseLoc+"errors.csv")
    distReports(ne).to_csv(baseLoc + "noterrors.csv")
    distReports(testSample).to_csv(baseLoc+"distTest.csv")

a = model.predict(a.transform(testSample))
predicted = testSample[[]]
predicted[target]=a
predicted.index.name=pk
predicted=predicted[[target]]
predicted.to_csv(baseLoc+"predicted.csv")

if False:
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
    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.barh(goodVar, goodImp)

    pyplot.show()
