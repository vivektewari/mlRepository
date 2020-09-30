from projectManager import  projectOwner
from dataManager import dataOwner,dataObject
from varManager import varOwner,varFactory
from varclushi import VarClusHi
from iv import IV
import pandas as pd
import warnings
warnings.filterwarnings ("ignore")
baseLoc='/home/pooja/PycharmProjects/homeCredit/tests2/'
stage=11

if stage==0:
    projectMan=projectOwner(loc='/home/pooja/PycharmProjects/homeCredit/')
    projectMan.initializeFolders()
if stage>3:
    folder='train/cleaned/'
    dataMan = dataOwner(loc=baseLoc + 'dataManagerFiles/', pk='SK_ID_CURR')
    varMan = varOwner(loc=baseLoc + 'varManagerFiles/', pk='SK_ID_CURR')


if stage==1: #trainValidSplit
    main=dataObject(loc=baseLoc+'baseDatasets/',name='target')
    main.load()
    dataObject.trainValidSplit(df=main.df,loc=main.loc,trainSize=0.8)
elif stage==2:#get the data files and user will define train test and valid
    dataMan=dataOwner(loc=baseLoc+'dataManagerFiles/', pk='SK_ID_CURR')
    dataMan.addDatacards( baseLoc+'baseDatasets/')

elif stage == 3:#suset dataset for all the datafile basis of it belongs to test,valid,train and puts it in required folder.
    dataMan=dataOwner(loc=baseLoc+'dataManagerFiles/', pk='SK_ID_CURR')
    dataMan.load()
    dataMan.getRelevantData()
elif stage == 4: #changed folder for data,get datacards ,user to feed primary ,rollup key

    dataMan.addDatacards(baseLoc + 'dataManagerFiles/'+folder)

elif stage ==5:#get initial reports for data cleaning; data cleaning is done on kaggle interface
    dataMan.load()
    dataMan.getInitialReports()
elif stage == 6:#produce the order of vars
    dataMan.load()

    varMan.addVarFromDataObjects(dataMan.dataCards)

    varMan.saveVarcards()
elif stage == 7:#produce the order of vars
    varMan.load()

    varMan.addVarFromDataObjects(dataMan.dataCards)

    varMan.saveVarcards()
elif stage == 8:# getting initial reorts for data cleaning
    dataMan.load()
    varMan.addVarFromDataObjects(dataMan.dataCards)
    #varMan.saveVarcards()
    varMan.addCoVar(dataMan.dataCards)
    varMan.saveVarcards()


elif stage == 9:# factory produce the vars
    varMan.load()
    dataMan.load()
    dataMan.dataCards[2].load()
    factoryMan=varFactory(varMan.getVarDF(),dataMan.dataCards,diag=1,target=dataMan.dataCards[2].df,pk='SK_ID_CURR',targetCol='TARGET',batchSize=10)
    factoryMan.produceVar()
elif stage==10: #getting the selected variable
    select=pd.read_csv('/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/report1.csv')
    select=select[select['select']==1].rename(columns={'varName':'name','code':'transformed'})
    select=select.drop_duplicates(['name','transformed'])
    varMan.load()
    storeCard=varMan.getVarDF()
    d=set(select.name).difference(set(storeCard.name))

    finalStoreCard=pd.merge(storeCard,select, on=['name','transformed'],how='inner')
    finalStoreCard.to_csv('/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/sel.csv')
elif stage==11:#produce the selected variable
    dataMan.load()
    dataMan.dataCards[2].load()
    finalStoreCard=pd.read_csv('/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/sel.csv')
    factoryMan = varFactory(finalStoreCard, dataMan.dataCards, diag=0, target=dataMan.dataCards[2].df,
                            pk='SK_ID_CURR',targetCol='TARGET')
    R=factoryMan.produceVar(indexes=dataMan.dataCards[2].df['SK_ID_CURR'][0:100])
    R.to_csv('/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/seldata.csv')

elif stage==12:#try varcluss
    data=pd.read_csv('/home/pooja/PycharmProjects/datanalysis/feature_cross/train_woe.csv')
    data=data.drop('trainBinned.csv',axis=1)
    # a=IV()
    # b=a.iv_all(data,target='TARGET')
    # c=a.convertToWoe(data,target='TARGET')
    demo1_vc = VarClusHi(data, maxeigval2=1, maxclus=None)
    demo1_vc.varclus(speedup=False)
    f=demo1_vc.info
    demo1_vc.rsquare.to_csv(baseLoc+"varluss.csv")
    #demo1_vc.info