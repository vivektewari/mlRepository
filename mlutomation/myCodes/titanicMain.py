#from projectManager import  projectOwner
from dataExploration import distReports,plotGrabh
from dataManager import dataOwner,dataObject
from varManager import varOwner,varFactory
# from varclushi import VarClusHi
from iv import IV
import pandas as pd
import warnings
warnings.filterwarnings ("ignore")
baseLoc='/home/pooja/PycharmProjects/titanic/'
stage=9#1
target='Survived'
pk='PassengerId'
if stage==0:
    projectMan=projectOwner(loc='/home/pooja/PycharmProjects/titanic/')
    projectMan.initializeFolders()
if stage>3:
    folder='train/'
    dataMan = dataOwner(loc=baseLoc + 'dataManagerFiles/', pk=pk,targetVar=target)
    varMan = varOwner(loc=baseLoc + 'varManagerFiles/', pk=pk)


if stage==1: #trainValidSplit
    dataMan = dataOwner(loc=baseLoc , pk=pk, targetVar=target)
    dataMan.makeTargetFile('main.csv')
    main=dataObject(loc=baseLoc+'baseDatasets/',name='target')
    main.load()
    dataObject.trainValidSplit(df=main.df,loc=main.loc,trainSize=0.8)
elif stage==2:#get the data files and user will define train test and valid
    folder = 'train/'
    dataMan=dataOwner(loc=baseLoc+'dataManagerFiles/', pk=pk)
    dataMan.addDatacards( dataMan.loc+folder)
    dataMan.save()
elif stage == 3:#suset dataset for all the datafile basis of it belongs to test,valid,train and puts it in required folder.
    dataMan=dataOwner(loc=baseLoc+'dataManagerFiles/', pk=pk)
    dataMan.load()
    dataMan.getRelevantData()
elif stage == 4: #changed folder for data,get datacards ,user to feed primary ,rollup key

    dataMan.addDatacards(baseLoc + 'baseDatasets/')
    dataMan.addParamsToCards()
    dataMan.save()

elif stage ==5:#get initial reports for data cleaning; data cleaning is done on kaggle interface
    dataMan.load()
    dataMan.getInitialReports()
    #df=pd.read_csv(baseLoc+'baseDatasets/main.csv')
    #plotGrabh(df,target=target,location=baseLoc+'baseDatasets/graphs/')
elif stage == 6:#produce the order of vars
    dataMan.addDatacards(dataMan.loc + folder)
    dataMan.addParamsToCards()
    dataMan.save()

elif stage == 7:# getting initial reorts for data cleaning

    dataMan.load()
    varMan.addVarFromDataObjects(dataMan.cards)
    #varMan.load()
    varMan.addCoVar(dataMan.cards)
    varMan.saveVarcards()


elif stage == 9:# factory produce the vars
    varMan.load()
    dataMan.load()
    dataMan.cards[1].load()
    #target=None
    tar=dataMan.cards[1].df
    book=varMan.getVarDF()
    #book['source']='testWithDummies'
    factoryMan=varFactory(book, dataMan.cards, diag=1, target=tar, pk=pk, targetCol=target, batchSize=100,diagFile=dataMan.loc+folder+"IVreport.csv")
    f=factoryMan.produceVar()

    if f is not None:f.to_csv(dataMan.loc+folder+"/mainwithCovars.csv",index_label=pk)
elif stage==10: #getting the selected variable
    dataMan.load()
    dataMan.cards[1].load()
    tar = dataMan.cards[1].df
    f=pd.read_csv(dataMan.loc + folder + "/mainwithCovars.csv", index_col=pk)
    f=f.join(tar.set_index(pk))
    f2 = pd.read_csv(dataMan.loc + folder + "/testwithCovars.csv", index_col=pk)
    f2.columns = [name.replace("testWithDummies", "mainWithDummies") for name in f2.columns]
    a = IV(getWoe=1)
    binned = a.binning(f, target, maxobjectFeatures=300, varCatConvert=1)
    ivData = a.iv_all(binned, target)

    converted_train = a.convertToWoe(f, binningOnly=0)
    a.saveVarcards(baseLoc, 'final_woe')
    new = IV()
    new.load(baseLoc, 'final_woe')
    converted_train = new.convertToWoe(f, binningOnly=0)
    converted_train2 = new.convertToWoe(f2, binningOnly=0)

    converted_train.to_csv(baseLoc+"0mainwithCovars.csv")
    converted_train2.to_csv(baseLoc + "0testwithCovars.csv")

elif stage==11:#produce the selected variable
    dataMan.load()
    dataMan.cards[2].load()
    finalStoreCard=pd.read_csv('/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/sel.csv')
    factoryMan = varFactory(finalStoreCard, dataMan.cards, diag=0, target=dataMan.cards[2].df,
                            pk='SK_ID_CURR', targetCol=target, batchSize=50)
    R=factoryMan.produceVar(indexes=dataMan.cards[2].df['SK_ID_CURR'], loc='/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/temp/')
    print(R.shape)
    R.to_csv('/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/seldata.csv')

elif stage==12:#saving a binned version:
    a=IV(getWoe=1,verbose=1)
    dataMan.load()
    dataMan.cards[2].load()
    final = pd.read_csv('/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/comb1.csv') #added post run
    final = final[final['select2'] == 1]
    vars = list(final['Variable'])
    train = pd.read_csv('/home/pooja/PycharmProjects/homeCredit/dataManagerFiles/train/cleaned/seldataTrial.csv')
    train = train.set_index(dataMan.pk)
    train=train[vars]
    train=train.join(dataMan.cards[2].df.set_index(dataMan.pk))

    binned = a.binning(train, target, maxobjectFeatures=300, varCatConvert=1)
    ivData = a.iv_all(binned, target)

    converted_train = a.convertToWoe(train,binningOnly=0)
    a.saveVarcards("/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/",'final_woe')
    new=IV()
    new.load("/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/",'final_woe')
    converted_train2 = new.convertToWoe(train, binningOnly=0)
    converted_train.to_csv('/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/seldata_woeFiletred_ramJaney.csv')
    c = converted_train2.eq(converted_train2)
    d = c.all()
    d.to_csv('/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/truth.csv')
    # train = pd.read_csv("/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/seldata_woeFiletred.csv")  ####'/home/pooja/PycharmProjects/datanalysis/finalDatasets/final.csv')
    #
    # tar = pd.read_csv("/home/pooja/PycharmProjects/homeCredit/tests2/dataManagerFiles/train/target.csv")
    # train = train.set_index('SK_ID_CURR').join(tar.set_index('SK_ID_CURR')[[target]]).drop('Unnamed: 0', axis=1)
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
    # b=a.iv_all(data,target=target)
    # c=a.convertToWoe(data,target=target)

elif stage==15:
    final=pd.read_csv('/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/comb1.csv')
    final=final[final['select2']==1]
    vars=list(final['Variable'])
    train = pd.read_csv('/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/seldata_woe1.csv')
    train=train.set_index('SK_ID_CURR').drop('Unnamed: 0', axis=1)
    train=train[vars]
    dataMan.load()
    dataMan.cards[2].load()
    final=train.join(dataMan.cards[2].df.set_index(dataMan.pk))
    final.to_csv('/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/dataFormodel_withTarget.csv')

elif stage==16:#get the plots
    train=pd.read_csv("/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/seldata_woeFiletredWithTarget.csv")
    plotGrabh(train,target,'/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/graphs/')
elif stage ==17:#make a prediction
   pass



