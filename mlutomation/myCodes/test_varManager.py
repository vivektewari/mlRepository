from unittest import TestCase
from dataManager import dataOwner
from varManager import varFactory,varOwner
import pandas as pd
import numpy as np
from iv import IV
from dataExploration import distReports
class TestvarFactory(TestCase):

    # dataMan = dataOwner(loc=baseLoc + 'dataManagerFiles/', pk='SK_ID_CURR')
    # varMan = varOwner(loc=baseLoc + 'varManagerFiles/', pk='SK_ID_CURR')
    # varMan.load()
    # dataMan.load()
    # dataMan.dataCards[2].load()
    # testSample=varMan.getVarDF()[[]]
    # factoryMan = varFactory(testSample, dataMan.dataCards, diag=1, target=dataMan.dataCards[2].df,
    #                         pk='SK_ID_CURR', targetCol='TARGET', batchSize=10)


    # def test1(self):
    #     # tests the diagnostic output from var factory and comapring with the manually calculator ones
    #
    #
    #     baseLoc = '/home/pooja/PycharmProjects/homeCredit/dataManagerFiles/'
    #     folder = 'train/cleaned/'
    #     data = pd.read_csv(baseLoc + folder+"previous_application.csv")
    #     target= pd.read_csv(baseLoc + folder + "target.csv")
    #     target=target.set_index('SK_ID_CURR')
    #     data=data[['SK_ID_CURR','DAYS_FIRST_DUE','NAME_YIELD_GROUP___2','AMT_APPLICATION','CNT_PAYMENT']]
    #     #data['NAME_YIELD_GROUP___2']=1
    #     data['DAYS_FIRST_DUE|NAME_YIELD_GROUP___2']=data['NAME_YIELD_GROUP___2'].div(data['DAYS_FIRST_DUE'], axis=0)
    #     data['NAME_YIELD_GROUP___2|DAYS_FIRST_DUE'] = data['DAYS_FIRST_DUE'].div(data['NAME_YIELD_GROUP___2'], axis=0)
    #     data['AMT_APPLICATION|CNT_PAYMENT']=data['AMT_APPLICATION']/data['CNT_PAYMENT']
    #
    #     d1=data.groupby('SK_ID_CURR')["DAYS_FIRST_DUE|NAME_YIELD_GROUP___2",'NAME_YIELD_GROUP___2|DAYS_FIRST_DUE','NAME_YIELD_GROUP___2'].mean()
    #     d2=data.groupby('SK_ID_CURR')["AMT_APPLICATION|CNT_PAYMENT","CNT_PAYMENT"].min()
    #     #d3 = data.groupby('SK_ID_CURR')["DAYS_FIRST_DUE|NAME_YIELD_GROUP___2", 'NAME_YIELD_GROUP___2'].sum().join(target)
    #     d=pd.concat([d1,d2],axis=1)
    #     d=d.join(target)
    #     col = 'TARGET'
    #     a = IV()
    #     final = distReports(d.drop(col, axis=1))
    #     binned = a.binning(d, col, maxobjectFeatures=300, varCatConvert=1)
    #     result = a.iv_all(binned, col)
    #     final1 = result.groupby('variable')['IV'].sum()
    #     final1 = pd.DataFrame(final1)
    #     final = final.join(final1)
    #     print(final)
    #     print(final.loc["DAYS_FIRST_DUE|NAME_YIELD_GROUP___2",'IV'])
    #     self.assertTrue(all([final.loc["DAYS_FIRST_DUE|NAME_YIELD_GROUP___2",'IV'] -0.114210662821109<0.001]), msg='may Be dataset object is not formed')

    def test2(self):
        # proding var from var factory and then matching with manually calculated one

        baseLoc = '/home/pooja/PycharmProjects/homeCredit/dataManagerFiles/'
        folder = 'train/cleaned/'
        data = pd.read_csv(baseLoc + folder+"previous_application.csv")
        target= pd.read_csv(baseLoc + folder + "target.csv")
        target=target.set_index('SK_ID_CURR')
        data=data[['SK_ID_CURR','DAYS_FIRST_DUE','NAME_YIELD_GROUP___2','NAME_YIELD_GROUP___1','AMT_APPLICATION','CNT_PAYMENT','DAYS_TERMINATION']]
        data=data.set_index('SK_ID_CURR')
        data=data[data.index.isin(target.index)]
        #data['NAME_YIELD_GROUP___2']=1
        data['DAYS_FIRST_DUE|NAME_YIELD_GROUP___2']=data['NAME_YIELD_GROUP___2'].div(data['DAYS_FIRST_DUE'], axis=0)
        data['NAME_YIELD_GROUP___1|DAYS_TERMINATION'] = data['DAYS_TERMINATION'].div(data['NAME_YIELD_GROUP___1'], axis=0)
        data['AMT_APPLICATION|CNT_PAYMENT']=data['AMT_APPLICATION']/data['CNT_PAYMENT']

        d1=data.groupby(data.index)['NAME_YIELD_GROUP___1|DAYS_TERMINATION','NAME_YIELD_GROUP___2'].max()
        d2=data.groupby(data.index)["AMT_APPLICATION|CNT_PAYMENT","CNT_PAYMENT"].min()
        #d3 = data.groupby('SK_ID_CURR')["DAYS_FIRST_DUE|NAME_YIELD_GROUP___2", 'NAME_YIELD_GROUP___2'].sum().join(target)
        d=pd.concat([d1,d2],axis=1)
        d = d.join(target)
        col = 'TARGET'
        varList1 = ['NAME_YIELD_GROUP___1|DAYS_TERMINATIONdiv;rollup,SK_ID_CURR,maxprevious_application']
        varList2 = ['NAME_YIELD_GROUP___1|DAYS_TERMINATION']
        testCandidate=pd.read_csv('./rest.csv',usecols=['NAME_YIELD_GROUP___1|DAYS_TERMINATIONdiv;rollup,SK_ID_CURR,maxprevious_application','SK_ID_CURR'],index_col='SK_ID_CURR')
        #testCandidate=testCandidate.set_index('SK_ID_CURR')[varList1]
        testCandidate=testCandidate.loc[list(set(testCandidate.index).intersection(d.index))]
        d=d.loc[list(set(testCandidate.index).intersection(d.index))]
        joined=testCandidate.join(d)

        g=joined[varList1[0]]-joined[varList2[0]]
        print(g.replace([np.inf,-np.inf,np.nan],0).sum())
        col = 'TARGET'
        a = IV()
        final = distReports(d.drop(col, axis=1))
        binned = a.binning(d, col, maxobjectFeatures=300, varCatConvert=1)
        result = a.iv_all(binned, col)
        final1 = result.groupby('variable')['IV'].sum()
        final1 = pd.DataFrame(final1)
        final = final.join(final1)
        print(final)
        #for v in varList
        #self.assertTrue(all([final.loc["DAYS_FIRST_DUE|NAME_YIELD_GROUP___2",'IV'] -0.114210662821109<0.001]), msg='may Be dataset object is not formed')

