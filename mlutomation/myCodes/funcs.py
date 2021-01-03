import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.metrics import roc_curve, auc
from multiprocessing import Pool, Process, cpu_count, Manager
from iv import IV
import time
import random
from os import path
import gc
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
def RMSE(dataset,actual,predicted):
    return (((dataset[actual]-dataset[predicted])**2).mean())**0.5
def logLoss(dataset,actual,predicted):
    dataset['error']=dataset[actual]*np.log(dataset[predicted]) + (1-dataset[actual])*np.log(1-dataset[predicted])
    return dataset
def deliquency(df1,rol,del_Var,monthVar, monthList=[12, 36, 120]):
    df = df1.copy()
    df_uniques = df.sort_values(['SK_ID_PREV', 'MONTHS_BALANCE'], ascending=False).drop_duplicates(
        subset=['SK_ID_PREV']) #taking latest information
    #df_uniques=df_uniques.drop(['NAME_CONTRACT_STATUS'], axis=1).set_index('SK_ID_PREV')
    #dfactives = df[df['NAME_CONTRACT_STATUS'] == "Active"].groupby(['SK_ID_PREV'])['NAME_CONTRACT_STATUS'].count()
    monthVarlist=[]
    for month in monthList:
        df["dpd_last_months_" + str(month)] = df.apply(
            lambda row: (1 + int(row[del_Var]) / 30) if (row[monthVar] > -month and row[del_Var] > 0) else 0,
            axis=1)
        monthVarlist.append("dpd_last_months_" + str(month))

    df2 = df.groupby(monthVarlist).max()
    final = df_uniques.join([df2])

    return final.reset_index()

def normalize(train,test=None):
        #l=train.mean()
        normalized_train=(train-train.mean())/train.std()
        if test is not None:
            normalized_test = (test - train.mean()) / train.std()
            return normalized_train,normalized_test
        return normalized_train


def aggregation(df1, rollupKey ='', monthVar='', aggFunc={}, monthList=[12, 36, 120]):#monh closeset is nbiggest
    df = df1.copy()

    df_uniques = df.sort_values([rollupKey, monthVar], ascending=False).drop_duplicates(
        subset=[rollupKey])  # taking latest information
    for month in monthList:
        df_temp = df[df[monthVar] > month]
        df_temp = df_temp.groupby([rollupKey]).agg(aggFunc)
        df_temp.columns = [str('_'+str(month)+"_").join(col).strip() for col in df_temp.columns.values]
        df_uniques=df_uniques.join(df_temp,on=[rollupKey])

    return df_uniques.fillna(0)
def crossVariable(df1,combination=None,target=None,varlist=None,ignoreList=None,numVar=0,memmorSaving=1):
    if varlist==None:varlist=df1.columns
    elif ignoreList==None:varlist=list(set(df1.columns)-set(ignoreList))
    df=df1[[]]
    if memmorSaving != 1: df=df1.drop(target,axis=1)
    if combination is None:combs=combinations(varlist, 2)
    else:combs=combination
    for comb in combs:
        if numVar !=1:
            df[comb[0] + "_&_" + comb[1] ] = df1[comb[0]] + "_&_"+df1[comb[1]]
        elif numVar==1:
            try:
                df[comb[0]+"_&_"+comb[1]+"m"]=df1[comb[0]]*df1[comb[1]]
            except TypeError:
                print(comb[0]+"_&_"+comb[1]+"m")
                pass
            try:
                df[comb[0] + "_&_"+ comb[1] + "d"] = df1[comb[0]] / df1[comb[1]]
            except ZeroDivisionError:
                df[comb[0] + "_&_" + comb[1] + "d"]=np.nan
            except TypeError:
                print(comb[0] + "_&_" + comb[1] + "m")
                pass

    if target is not None:df[target]=df1[target]
    return df #.replace(np.inf,np.nan)
#low ram

def getIVForCross(df1, combination, target=None, number=None, loc=None,groupByKey=None):

    a=IV
    if groupByKey is not None:
        temp = crossVariable(df1, combination, target=target, varlist=None, ignoreList=None,numVar=1,memmorSaving=0)
        #temp=temp.fillna(-579579)
        temp=temp.groupby(groupByKey).agg(['min', 'max','sum','mean'])
        temp.columns = [str("_").join(col).strip() for col in temp.columns.values]
        temp=temp.drop([target+"_"+ f for f in ['min', 'max','sum']],axis=1)
        temp.rename(columns = {target+"_mean":target}, inplace = True)
        binned = a.binning(temp, target,qCut=10, maxobjectFeatures=50,varCatConvert=1)
        ivData = a.iv_all(binned, target,modeBinary=0)
    else:
        temp = crossVariable(df1, combination, target=target, varlist=None, ignoreList=None)
        ivData = a.iv_all(temp, target,modeBinary=0)
    ivData.groupby('variable')['ivValue'].sum().to_csv(loc+str(number)+".csv",mode = 'a', header = False)

def crossVariablelowRam(df1,train=None,varlist=None,ignoreList=None,target=None,batch=10,loc=None,groupByKey=None):
    """

    :param df1: dataframe|df for which cross variable to be calculated
    :param train: dataframe|dataframe containing target variable. Only needed if IV needs to be calculated
    :param varlist: string list|variables in df1 for which cross will be calculated. If None then all variables of dataframe will be considered
    :param ignoreList:string list| varibles which will not be considered fr cross vars
    :param target: string|target variable
    :param batch: int|for parallel processing how many combination will be parrallely processed
    :param loc: string|where all the outputs will be made
    :param groupByKey: string| key with which groupBy to be done before IV calculation .Only applicable for numeric variable
    :return: CSV file| a file consisting of all the cross variable and its IV value
    """
    start = time.time()
    outputFile=pd.DataFrame(columns =['ivValue'])
    for i in range(batch):outputFile.to_csv(loc+str(i)+".csv")
    cores=cpu_count()
    pool = Pool(processes=cores)
    if varlist is None:varlist=df1.columns
    if ignoreList is not None:varlist=list(set(df1.columns)-set(ignoreList))
    coreNum=0
    excludes=[]
    if groupByKey is None:
        binned = binning(df1,  qCut=10, maxobjectFeatures=50,varCatConvert=1,excludedList=excludes)
        varlist=list(set(varlist)-set(excludes))
        binned = binned.astype(str)
        binned.columns = [col.replace('n_', "").replace('c_', "") for col in binned.columns]

    else:
        objectCols = list(df1.select_dtypes(include=['object']).columns)
        varlist = list(set(varlist) - set(objectCols))
        binned=df1[varlist]
    combs = combinations(varlist, 2)
    total_cross = len(list(combs))
    numberOfBatches = int(total_cross / batch)
    binned=binned.join(train[[target]])
    #print(binned.dtypes)
    print(numberOfBatches)
    combs = list(combinations(varlist, 2))
    i=0
    for i in range(0,numberOfBatches):
        cross=combs[i*batch:i*batch+batch]
        vars=list(set([com[0] for com in cross]).union(set([com[1] for com in cross])))+[target]
        pool.apply_async(getIVForCross, args=(binned[vars], cross, target, i % batch, loc,groupByKey ))
        #print(i)
        coreNum=coreNum+1
        #gc.collect()
        if i%int(batch/2)==0:#used so that limited process run in memmory. So batch should be cosen considering availibilty of memmory
            pool.close()
            pool.join()
            print(i)
            pool = Pool(processes=cores)


    cross = combs[i * batch:total_cross]
    vars = list(set([com[0] for com in cross]).union(set([com[1] for com in cross]))) + [target]
    pool.apply_async(getIVForCross, args=(binned[vars], cross, target, i % batch, loc,groupByKey))

    pool.close()
    pool.join()
    for i in range(batch):
        if i == 0:
            main = pd.read_csv(loc + str(i) + ".csv")
            main.to_csv(loc + ".csv")
        else:
            main = pd.read_csv(loc+ str(i) + ".csv")
            main.to_csv(loc+ "t.csv", mode='a',header=False)
    end = time.time()
    print("total Time taken" +str((end - start)/60))


def isPrimaryKey(df,varList):
    """

    :param df:dataframe| for which we are checking primary key
    :param varList: list| varList by which primary key can be formed
    :return: boolean| True if yes
    """
    df['pk']=""
    for var in varList:
        df['pk']=df['pk']+df[var].map(str)

    return df.shape[0]==len(df['pk'].unique())

def lorenzCurve(y_test,y_score):
    n_classes = 1
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _= roc_curve(y_test, y_score)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    #Plot of a ROC curve for a specific class

    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

def getMean(df,var1,target):

    main=df
    for var in var1:
       a= df[[var,target]].dropna(subset=[target]).groupby(var)[target].agg(['mean']).rename(columns={'mean':var+'MeanT'})
       main=main.join(a,on=var)
       # main[var+'MeanT']=main.apply(lambda row:row[var+'MeanT'] if np.isnan(row[target]) else (np.nan if (row[var+'CountT']==1 or row[var+'CountT']==np.nan) else (row[var+'MeanT']*row[var+'CountT']-row[target])/(row[var+'CountT']-1)),axis=1)
       # main=main.drop(var+'CountT',axis=1)
    return main
def getCount(df,var1):

    main=df
    for var in var1:
       a= df[[var]].groupby(var)[var].agg(['count']).rename(columns={'count':var+'Count'})
       main=main.join(a,on=var)
    return main

def stratified_sample(df, strata, size=None, seed=None, keep_index=True):
    '''
    It samples data from a pandas dataframe using strata. These functions use
    proportionate stratification:
    n1 = (N1/N) * n
    where:
        - n1 is the sample size of stratum 1
        - N1 is the population size of stratum 1
        - N is the total population size
        - n is the sampling size
    Parameters
    ----------
    :df: pandas dataframe from which data will be sampled.
    :strata: list containing columns that will be used in the stratified sampling.
    :size: sampling size. If not informed, a sampling size will be calculated
        using Cochran adjusted sampling formula:
        cochran_n = (Z**2 * p * q) /e**2
        where:
            - Z is the z-value. In this case we use 1.96 representing 95%
            - p is the estimated proportion of the population which has an
                attribute. In this case we use 0.5
            - q is 1-p
            - e is the margin of error
        This formula is adjusted as follows:
        adjusted_cochran = cochran_n / 1+((cochran_n -1)/N)
        where:
            - cochran_n = result of the previous formula
            - N is the population size
    :seed: sampling seed
    :keep_index: if True, it keeps a column with the original population index indicator

    Returns
    -------
    A sampled pandas dataframe based in a set of strata.
    Examples
    --------
    >> df.head()
    	id  sex age city
    0	123 M   20  XYZ
    1	456 M   25  XYZ
    2	789 M   21  YZX
    3	987 F   40  ZXY
    4	654 M   45  ZXY
    ...
    # This returns a sample stratified by sex and city containing 30% of the size of
    # the original data
    >> stratified = stratified_sample(df=df, strata=['sex', 'city'], size=0.3)
    Requirements
    ------------
    - pandas
    - numpy
    '''
    population = len(df)
    size = __smpl_size(population, size)
    tmp = df[strata]
    tmp['size'] = 1
    tmp_grpd = tmp.groupby(strata).count().reset_index()
    tmp_grpd['samp_size'] = round(size / population * tmp_grpd['size']).astype(int)

    # controlling variable to create the dataframe or append to it
    first = True
    for i in range(len(tmp_grpd)):
        # query generator for each iteration
        qry = ''
        for s in range(len(strata)):
            stratum = strata[s]
            value = tmp_grpd.iloc[i][stratum]
            n = tmp_grpd.iloc[i]['samp_size']

            if type(value) == str:
                value = "'" + str(value) + "'"

            if s != len(strata) - 1:
                qry = qry + stratum + ' == ' + str(value) + ' & '
            else:
                qry = qry + stratum + ' == ' + str(value)

        # final dataframe
        if first:
            stratified_df = df.query(qry).sample(n=n, random_state=seed).reset_index(drop=(not keep_index))
            first = False
        else:
            tmp_df = df.query(qry).sample(n=n, random_state=seed).reset_index(drop=(not keep_index))
            stratified_df = stratified_df.append(tmp_df, ignore_index=True)

    return stratified_df


def stratified_sample_report(df, strata, size=None):
    '''
    Generates a dataframe reporting the counts in each stratum and the counts
    for the final sampled dataframe.
    Parameters
    ----------
    :df: pandas dataframe from which data will be sampled.
    :strata: list containing columns that will be used in the stratified sampling.
    :size: sampling size. If not informed, a sampling size will be calculated
        using Cochran adjusted sampling formula:
        cochran_n = (Z**2 * p * q) /e**2
        where:
            - Z is the z-value. In this case we use 1.96 representing 95%
            - p is the estimated proportion of the population which has an
                attribute. In this case we use 0.5
            - q is 1-p
            - e is the margin of error
        This formula is adjusted as follows:
        adjusted_cochran = cochran_n / 1+((cochran_n -1)/N)
        where:
            - cochran_n = result of the previous formula
            - N is the population size
    Returns
    -------
    A dataframe reporting the counts in each stratum and the counts
    for the final sampled dataframe.
    '''
    population = len(df)
    size = __smpl_size(population, size)
    tmp = df[strata]
    tmp['size'] = 1
    tmp_grpd = tmp.groupby(strata).count().reset_index()
    tmp_grpd['samp_size'] = round(size / population * tmp_grpd['size']).astype(int)
    return tmp_grpd


def __smpl_size(population, size):
    '''
    A function to compute the sample size. If not informed, a sampling
    size will be calculated using Cochran adjusted sampling formula:
        cochran_n = (Z**2 * p * q) /e**2
        where:
            - Z is the z-value. In this case we use 1.96 representing 95%
            - p is the estimated proportion of the population which has an
                attribute. In this case we use 0.5
            - q is 1-p
            - e is the margin of error
        This formula is adjusted as follows:
        adjusted_cochran = cochran_n / 1+((cochran_n -1)/N)
        where:
            - cochran_n = result of the previous formula
            - N is the population size
    Parameters
    ----------
        :population: population size
        :size: sample size (default = None)
    Returns
    -------
    Calculated sample size to be used in the functions:
        - stratified_sample
        - stratified_sample_report
    '''
    if size is None:
        cochran_n = round(((1.96) ** 2 * 0.5 * 0.5) / 0.02 ** 2)
        n = round(cochran_n / (1 + ((cochran_n - 1) / population)))
    elif size >= 0 and size < 1:
        n = round(population * size)
    elif size < 0:
        raise ValueError('Parameter "size" must be an integer or a proportion between 0 and 0.99.')
    elif size >= 1:
        n = size
    return n
