from sklearn.preprocessing import StandardScaler
import warnings
import time
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
from funcs import lorenzCurve,normalize,logLoss
from statsmodels.tools.tools import add_constant
start = time.time()
warnings.filterwarnings("ignore")
from multiprocessing import Pool, Process, cpu_count, Manager
from sklearn import metrics
method='NN'
case=2
pk='PassengerId'
baseLoc='/home/pooja/PycharmProjects/titanic/'

target='Survived'

pk='PassengerId'
main=pd.read_csv(baseLoc+'dataManagerFiles/train/'+"mainwithCovars.csv",index_col=pk)
tar= pd.read_csv(baseLoc + 'dataManagerFiles/train/' + "target.csv", index_col=pk)
main=main.join(tar)
testSample = pd.read_csv(baseLoc + 'dataManagerFiles/train/' + "testwithCovars.csv",index_col=pk)
testSample.columns=[name.replace("testWithDummies","mainWithDummies") for name in testSample.columns]

main=main.fillna(main.mean())
# maxi=main.max()
# mini=main.min()
# diff=pd.DataFrame(index=main.coulmns,values=maxi-mini)
# zeroes=diff[diff[0]==0].columns




testSample=testSample.fillna(main.mean())
main ,testSample= normalize(main.drop(target,axis=1), testSample)
nanColumns=main.columns[main.isna().any()].tolist()
main=main.drop(nanColumns,axis=1)
testSample=testSample.drop(nanColumns,axis=1)
main = main.join(tar)
test=main.sample(n=int(main.shape[0]*0.20) ,random_state=0)
train=main.drop(test.index,axis=0)
test_y=test[[target]]
train_y=train[[target]]
train=train.drop(target,axis=1)
test=test.drop(target,axis=1)
X=train
y=train_y


varSelected=['Age0mainWithDummies', 'TicketMeanT0mainWithDummies', 'Age|FaredivmainWithDummies', 'Age|SibSpdivmainWithDummies', 'Age|PclassdivmainWithDummies', 'Age|SexdivmainWithDummies', 'Age|cabinConvertMeanTdivmainWithDummies', 'Fare|cabinConvertdivmainWithDummies', 'Fare|SibSpdivmainWithDummies', 'Fare|PclassdivmainWithDummies', 'Fare|ParchdivmainWithDummies', 'Fare|Embarked_SdivmainWithDummies', 'Fare|SexdivmainWithDummies', 'cabinConvert|SexdivmainWithDummies', 'SibSp|SexdivmainWithDummies', 'SibSp|cabinConvertMeanTdivmainWithDummies', 'Pclass|cabinConvertMeanTdivmainWithDummies', 'Embarked_C|TicketMeanTdivmainWithDummies', 'Age|FaremultmainWithDummies', 'Age|SibSpmultmainWithDummies', 'Age|PclassmultmainWithDummies', 'Age|SexmultmainWithDummies', 'Fare|SibSpmultmainWithDummies', 'Fare|PclassmultmainWithDummies', 'Fare|Embarked_SmultmainWithDummies', 'Fare|cabinConvertMeanTmultmainWithDummies', 'cabinConvert|SexmultmainWithDummies', 'SibSp|PclassmultmainWithDummies', 'Pclass|ParchmultmainWithDummies']
varSelected=X.columns


def uplevel(mlp, train, trainTarget, i, j):
    mlp.fit(train, trainTarget[target])
    return [mlp, i, j]


def jugad(que, mlp, train, trainTarget, i, j):
    que.put(uplevel(mlp, train, trainTarget, i, j))


def runModel(train, test, trainTarget, testTarget, targetVar=target, subMode=None,submission=0):
    print("started")
    from sklearn.neural_network import MLPClassifier
    if subMode is None:
        que = Manager().Queue()
        pool = Pool(processes=1)
        counter = 0

    dict1 = {}
    L1 = [i for i in range(12, 20)]
    L2 = [i for i in range(4, 12)]
    for i in L1:
        for j in L2:
            if j < i and subMode is None:
                if j != 0:
                    mlp = MLPClassifier(hidden_layer_sizes=(i, j), max_iter=250, alpha=1e-4,
                                        solver='sgd', verbose=False, tol=1e-4, random_state=1,
                                        learning_rate_init=0.1)
                else:
                    mlp = MLPClassifier(hidden_layer_sizes=(i), max_iter=250, alpha=1e-4,
                                        solver='sgd', verbose=False, tol=1e-4, random_state=1,
                                        learning_rate_init=0.1)
                pool.apply_async(jugad, args=(que, mlp, train, trainTarget, i, j))
                counter = counter + 1
            elif subMode is not None:
                mlp = MLPClassifier(hidden_layer_sizes=subMode, max_iter=250, alpha=1e-4,
                                    solver='sgd', verbose=False, tol=1e-4, random_state=1,
                                    learning_rate_init=0.1)
                mlp.fit(train, trainTarget[target])
                trainer = pd.DataFrame(mlp.predict_proba(train.values), columns=['good', target], index=train.index)[
                    [target]]
                submision = pd.DataFrame(mlp.predict_proba(test.values), columns=['good', target], index=test.index)[
                    [target]]
                print(trainTarget[[target]].mean(axis=0), trainer[[target]].mean(axis=0))
                #lorenzCurve(trainTarget["TARGET"].values.flatten(),trainer[target].values.flatten())
                # rawTest['actual'] = testTarget[target]
                # rawTest['TARGET1']=submision[target]
                score_test = metrics.roc_auc_score(testTarget[target], submision[[target]])
                score_train = metrics.roc_auc_score(trainTarget[target], trainer[[target]])
                print(score_train,score_test)
                #error = logLoss(rawTest[varSelected+['actual', 'TARGET1']], 'actual', 'TARGET1').sort_values(['error'])
                #error.to_csv("/home/pooja/PycharmProjects/datanalysis/finalDatasets/error.csv")
                return pd.DataFrame(mlp.predict_proba(test.values), columns=['good',target],
                                    index=testTarget.index)[[target]]

    pool.close()
    pool.join()
    for element in range(counter):
        field = que.get()
        mlp = field[0]
        i = field[1]
        j = field[2]
        trainer = pd.DataFrame(mlp.predict_proba(train.values), columns=['good', target], index=train.index)[
            [target]]
        submision = pd.DataFrame(mlp.predict_proba(test.values), columns=['good', target], index=test.index)[
            [target]]
        # submision.index.name='SK_ID_CURR'
        # print(submision.shape)
        # submision.to_csv("submission.csv")
        score_test = metrics.roc_auc_score(testTarget[target], submision[[target]])
        score_train = metrics.roc_auc_score(trainTarget[target], trainer[[target]])



        try:
            dict1[str(i) + "_" + str(j)] = str(score_train) + "_" + str(score_test)
        except:
            print(i, j)
            pass

    print("starting")
    for key in dict1.keys():
        print(key)
        print(dict1[key])
    return None
subCard=runModel(X,test,train_y,test_y)#,subMode=(16,8))
#subCard=runModel(train[varSelected].replace([np.inf,np.nan],0),test[varSelected].replace([np.inf,np.nan],0),train[[target]],test[[target]])#,subMode=(15,5)

#subCard=runModel(train_t,test_t,train[[target]],test[[target]],subMode=(13,4)
#subCard.to_csv("/home/pooja/PycharmProjects/datanalysis/finalDatasets/submission.csv")
end = time.time()

print(end - start)