from projectManager import  projectOwner
from commonFuncs import packing
from dataExploration import distReports,plotGrabh
from dataManager import dataOwner,dataObject
from varManager import varOwner,varFactory
from varclushi import VarClusHi
from iv import IV
import pandas as pd
import warnings
import time
import copy
start=time.time()
warnings.filterwarnings ("ignore")
baseLoc='/home/pooja/PycharmProjects/homeCredit/' #tests2/' #'/home/pooja/PycharmProjects/homeCredit/'
stage=12

if stage==0:
    projectMan=projectOwner(loc='/home/pooja/PycharmProjects/homeCredit/')
    projectMan.initializeFolders()
if stage>3:
    folder='/dataManagerFiles/test/cleaned/'
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
    dataMan.dataCards[7].load()
    factoryMan=varFactory(varMan.getVarDF(),dataMan.dataCards,diag=0,pk='SK_ID_CURR',targetCol='TARGET',batchSize=10)
    R=factoryMan.produceVar(indexes=dataMan.dataCards[7].df[dataMan.pk])
    print(R.shape)
    R.to_csv(dataMan.loc + folder + '/seldata.csv')
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
    varMan.load()
    dataMan.dataCards[2].load()


    factoryMan = varFactory(varMan.getVarDF(), dataMan.dataCards, diag=0, target=dataMan.dataCards[2].df,
                            pk='SK_ID_CURR',targetCol='TARGET',batchSize=20,train=0)

    a=IV(getWoe=1)
    a.load('/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/',name='ivResults')
    factoryMan.IVMan=a
    print("starting production")
    R=factoryMan.produceVar(indexes=dataMan.dataCards[2].df['SK_ID_CURR'],loc='/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/temp/')
    print(R.shape)
    R.to_csv(baseLoc +folder+'/seldataTrial.csv')

elif stage==12:#saving a binned version:
    a = IV(getWoe=1, verbose=1)
    dataMan.load()
    dataMan.dataCards[2].load()
    final = pd.read_csv('/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/comb1.csv')  # added post run
    final = final[final['select2'] == 1]
    vars = list(final['Variable'])
    train = pd.read_csv(baseLoc+folder+'/seldataTrial.csv')
    train = train.set_index(dataMan.pk)
    train = train[vars]
    #train = train.join(dataMan.dataCards[2].df.set_index(dataMan.pk))

    new = IV()
    new.load("/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/", 'final_woe')
    converted_train = new.convertToWoe(train, binningOnly=0)
    converted_train.to_csv(baseLoc +folder+'/seldata_woeFiletred_ramJaney.csv')
    # c = converted_train2.eq(converted_train2)
    # d = c.all()
    # d.to_csv('/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/truth.csv')
elif stage==13:
    pass
    a=IV(getWoe=1,verbose=1)
    dataMan.load()
    dataMan.dataCards[2].load()
    final = pd.read_csv('/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/comb1.csv') #added post run
    final = final[final['select2'] == 1]
    vars = list(final['Variable'])
    train = pd.read_csv('/home/pooja/PycharmProjects/homeCredit/dataManagerFiles/train/application_train.csv')
    train = train.set_index(dataMan.pk).drop('Unnamed: 0',axis=1)
    #train=train[vars]
    train=train.join(dataMan.dataCards[2].df.set_index(dataMan.pk))#joining target

    binned = a.binning(train, 'TARGET', maxobjectFeatures=300, varCatConvert=1)
    c1 = copy.deepcopy(a.variables)
    a.saveVarcards('/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/',name='ivResults')
    a = IV(getWoe=1, verbose=1)
    a.load('/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/',name='ivResults')
    binned2=a.convertToWoe(train,'TARGET',binningOnly=1)
    c=binned.eq(binned2)
    d=c.all()
    d.to_csv('/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/truth.csv')
elif stage == 99:pass
    #c=a.iv_all(binned,'TARGET')
    #c=copy.deepcopy(a.variables)
    #x=packing.strToList(a.variables[0].bins)
    # var1=list(c.values())
    # var2=list(a.variables.values())
    # print('s')
    # for i in range(len(var1)):
    #     #print(var1[i].name)
    #     t=True
    #     try:
    #         g = [list(var1[i].catDictionary.keys())[j] == list(var2[i].catDictionary.keys())[j] for j in
    #                 range(len(list(var2[i].catDictionary.keys())))]
    #         t=all(list(var1[i].catDictionary.values())[j]==list(var2[i].catDictionary.values())[j] for j in range(len(list(var2[i].catDictionary.keys()))))
    #     except:
    #         print(type(var1[i].catDictionary))
    #     if not t:
    #         print(var1[i].name)
            # for j in range(len(var1[i].bins)):
            #     print(var1[i].bins[j]==var2[i].bins[j])
    #ivData = a.iv_all(binned, 'TARGET')
    # for folder in ['valid/cleaned/']:#,'test/cleaned/']:
    #     candidate=pd.read_csv('/home/pooja/PycharmProjects/homeCredit/dataManagerFiles/'+folder+'seldata.csv',nrows=100)
    #     candidate['Unnamed: 0'] = 0
    #     converted_train = a.convertToWoe(candidate.set_index(dataMan.pk))
    #     converted_train.to_csv('/home/pooja/PycharmProjects/homeCredit/dataManagerFiles/'+folder+'seldata_woeFiletred2.csv')

    # train = pd.read_csv("/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/seldata_woeFiletred.csv")  ####'/home/pooja/PycharmProjects/datanalysis/finalDatasets/final.csv')
    #
    # tar = pd.read_csv("/home/pooja/PycharmProjects/homeCredit/tests2/dataManagerFiles/train/target.csv")
    # train = train.set_index('SK_ID_CURR').join(tar.set_index('SK_ID_CURR')[['TARGET']]).drop('Unnamed: 0', axis=1)
    # train.to_csv("/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/seldata_woeFiletredWithTarget.csv")
elif stage == 13:  # saving a binned version:
    converted_train=pd.read_csv('/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/seldata_woe1.csv')
    converted_train = converted_train.drop('Unnamed: 0', axis=1)
    #d=distReports(converted_train)

    #d.to_csv('/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/distr.csv')
    converted_train=converted_train.set_index('SK_ID_CURR')
    demo1_vc = VarClusHi(converted_train, maxeigval2=1, maxclus=None)
    demo1_vc.varclus(speedup=True)
    f = demo1_vc.info
    demo1_vc.rsquare.to_csv(baseLoc + "varluss.csv")

elif stage == 14:#mixing varcluss with iv report
    iv=pd.read_csv('/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/report1.csv')
    iv['Variable']=iv.apply(lambda row: row['varName']+row['code'],axis=1)
    varclus = pd.read_csv(baseLoc + "varluss.csv")
    final=varclus.set_index('Variable').join(iv.set_index('Variable'))
    final.to_csv('/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/comb1.csv')
    # a=IV()
    # b=a.iv_all(data,target='TARGET')
    # c=a.convertToWoe(data,target='TARGET')

elif stage==15:
    final=pd.read_csv('/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/comb1.csv')
    final=final[final['select2']==1]
    vars=list(final['Variable'])
    train = pd.read_csv('/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/seldata_woe1.csv')
    train=train.set_index('SK_ID_CURR').drop('Unnamed: 0', axis=1)
    train=train[vars]
    dataMan.load()
    dataMan.dataCards[2].load()
    final=train.join(dataMan.dataCards[2].df.set_index(dataMan.pk))
    final.to_csv('/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/dataFormodel_withTarget.csv')

elif stage==16:#get the plots
    train=pd.read_csv("/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/seldata_woeFiletredWithTarget.csv")
    plotGrabh(train,'TARGET','/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/graphs/')
elif stage ==17:#make a prediction
   pass

print(time.time()-start)

if __name__=='__main__':
    pass