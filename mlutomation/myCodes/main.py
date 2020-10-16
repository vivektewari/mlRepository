from projectManager import  projectOwner
from dataExploration import distReports,plotGrabh
from dataManager import dataOwner,dataObject
from varManager import varOwner,varFactory
from varclushi import VarClusHi
from iv import IV
import pandas as pd
import warnings
warnings.filterwarnings ("ignore")
baseLoc='/home/pooja/PycharmProjects/homeCredit/'
stage=19

if stage==0:
    projectMan=projectOwner(loc='/home/pooja/PycharmProjects/homeCredit/')
    projectMan.initializeFolders()
if stage>3:
    folder='test/cleaned/forModelRun/'
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

elif stage == 7:# getting initial reorts for data cleaning
    dataMan.load()
    #varMan.addVarFromDataObjects(dataMan.dataCards)
    #varMan.load()
    varMan.addCoVar(dataMan.dataCards)
    varMan.saveVarcards()


elif stage == 9:# factory produce the vars
    varMan.load()
    dataMan.load()
    dataMan.dataCards[2].load()
    factoryMan=varFactory(varMan.getVarDF(),dataMan.dataCards,diag=1,target=dataMan.dataCards[2].df,pk='SK_ID_CURR',targetCol='TARGET',batchSize=10)

    factoryMan.produceVar(indexes=dataMan.dataCards[2].df['SK_ID_CURR'])
elif stage==10: #getting the selected variable
    select=pd.read_csv("./fileterIv.csv")
    select=select.rename(columns={'varName':'name','code':'transformed'})[['name','transformed','source']].set_index(['name','transformed','source'])
    #select=select.drop_duplicates(['name','transformed'])
    varMan.load()
    storeCard=varMan.getVarDF()
    #d=set(select.name).difference(set(storeCard.name))

    finalStoreCard=pd.merge(storeCard,select, on=['name','transformed','source'],how='inner')
    finalStoreCard.to_csv('/home/pooja/PycharmProjects/homeCredit/varManagerFiles/book.csv',index=False)
elif stage==11:#produce the selected variable
    dataMan.load()
    dataMan.dataCards[2].load()
    varMan.load()
    #finalStoreCard=pd.read_csv('/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/sel.csv')
    factoryMan = varFactory(varMan.getVarDF(), dataMan.dataCards, diag=0, target=dataMan.dataCards[2].df,
                            pk='SK_ID_CURR',targetCol='TARGET',batchSize=50)
    R=factoryMan.produceVar(indexes=dataMan.dataCards[2].df['SK_ID_CURR'],loc='/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/temp/')
    print(R.shape)
    R.to_csv(baseLoc + 'dataManagerFiles/'+ folder+'rest.csv',index_label=dataMan.pk)

elif stage==12:#saving a binned version:
    a=IV(getWoe=1,verbose=1)
    dataMan.load()
    dataMan.dataCards[2].load()
    final = pd.read_csv('/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/comb1.csv') #added post run
    final = final[final['select2'] == 1]
    vars = list(final['Variable'])
    train = pd.read_csv('./dataFormodel_withTarget.csv')
    train = train.set_index(dataMan.pk)
    #train=train[vars]
    #train=train.join(dataMan.dataCards[2].df.set_index(dataMan.pk))

    binned = a.binning(train, 'TARGET', maxobjectFeatures=300, varCatConvert=1)
    ivData = a.iv_all(binned, 'TARGET')

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
    # train = train.set_index('SK_ID_CURR').join(tar.set_index('SK_ID_CURR')[['TARGET']]).drop('Unnamed: 0', axis=1)
    # train.to_csv("/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/seldata_woeFiletredWithTarget.csv")
elif stage==122:#binned version for rest of datasets and varcluss
    sLoc='/home/pooja/PycharmProjects/homeCredit/dataManagerFiles/train/cleaned/'
    dLoc='/home/pooja/PycharmProjects/homeCredit/dataManagerFiles/train/cleaned/forModelRun/'
    dataMan.load()
    dataMan.dataCards[2].load()
    datasetNames = ['previous_application.csv', 'POS_CASH_balance.csv', 'credit_card_balance.csv', 'bureau.csv']
    pk='SK_ID_CURR'
    varfinal = pd.read_csv("./fileterIv.csv")
    varfinal['varCode'] = varfinal['varName'] + varfinal['code'] + varfinal['source']
    # added post run
    for dataset in datasetNames:
        print(dataset)
        data = pd.read_csv(sLoc + dataset)
        relevantIndex = data[pk]
        varfinal1=varfinal[varfinal['source']==dataset.split(".csv")[0]]
        relevantVars=list(varfinal1['varCode'])+[pk]
        dfinal = pd.read_csv("./rest.csv",usecols=relevantVars)
        dfinal=dfinal[dfinal[pk].isin(relevantIndex) ]
        dfinal=dfinal.set_index(dataMan.pk).join(dataMan.dataCards[2].df.set_index(dataMan.pk))
        a = IV(getWoe=1, verbose=1)
        binned = a.binning(dfinal, 'TARGET', maxobjectFeatures=300, varCatConvert=1)
        ivData = a.iv_all(binned, 'TARGET')
        converted_train = a.convertToWoe(dfinal, binningOnly=0)
        VarClusHi(converted_train, maxeigval2=1, maxclus=None)
        demo1_vc.varclus(speedup=False)
        demo1_vc=demo1_vc.rsquare
        demo1_vc['source']=dataset
        if  dataset=='previous_application.csv':demo1_vc.to_csv(dLoc+ "varluss.csv")
        else :demo1_vc.to_csv(dLoc + "varluss.csv",mode='a')
    v=pd.read_csv(dLoc + "varluss1.csv",index_col='Variable')
    v=v.join(varfinal.set_index('varCode'),rsuffix="_")
    v.to_csv(dLoc + "varluss2.csv")





elif stage == 13:  # saving a binned version:
    converted_train=pd.read_csv('/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/seldata_woe1.csv',nrows=1000)
    converted_train = converted_train.drop('Unnamed: 0', axis=1)
    #d=distReports(converted_train)

    #d.to_csv('/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/distr.csv')
    converted_train=converted_train.set_index('SK_ID_CURR')
    demo1_vc = VarClusHi(converted_train, maxeigval2=1, maxclus=None)
    demo1_vc.varclus(speedup=False)
    f = demo1_vc.info
    demo1_vc.rsquare.to_csv(baseLoc + "varluss.csv")

elif stage == 14:#final sortlisting variable and changing the book for variable list
    sLoc='/home/pooja/PycharmProjects/homeCredit/dataManagerFiles/train/cleaned/'
    dLoc='/home/pooja/PycharmProjects/homeCredit/dataManagerFiles/train/cleaned/forModelRun/'
    varMan.load()
    varclus = pd.read_csv(dLoc + "varluss2.csv")
    initialBook=varMan.getVarDF()
    varclus=varclus[varclus['Select']==1]
    varclus=varclus[['varName','code','source_']].rename(columns={'varName':'name','code':'transformed','source_':'source'}).set_index(['name','transformed','source'])
    finalBook=initialBook.set_index(['name','transformed','source']).join(varclus,how='inner').reset_index()
    #varclus['Variable'] = varclus.apply(lambda row: row['varName'] + row['code'] + row['source'], axis=1)
    finalBook=dataObject(df=finalBook,name='book')
    varMan.load(temp=finalBook)
    varMan.saveVarcards()
    # a=IV()
    # b=a.iv_all(data,target='TARGET')
    # c=a.convertToWoe(data,target='TARGET')

elif stage==15:#filtering the final variable set
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
elif stage==16:
    sLoc = '/home/pooja/PycharmProjects/homeCredit/dataManagerFiles/train/cleaned/'
    dLoc = '/home/pooja/PycharmProjects/homeCredit/dataManagerFiles/train/cleaned/forModelRun/'
    dataMan.load()
    dataMan.dataCards[2].load()
    varMan.load()
    finalStoreCard=pd.read_csv('/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/sel.csv')
    factoryMan = varFactory(varMan.getVarDF(), dataMan.dataCards, diag=0, target=dataMan.dataCards[2].df,
                            pk='SK_ID_CURR', targetCol='TARGET', batchSize=50)
    R = factoryMan.produceVar(indexes=dataMan.dataCards[2].df['SK_ID_CURR'],
                              loc='/home/pooja/PycharmProjects/pythonProject/mlutomation/myCodes/temp/')
    R.to_csv(dLoc+'/rest.csv', index_label=dataMan.pk)



    output=dataMan.dataCards[2].df.set_index(dataMan.pk)[[]]
    varfinal=varMan.getVarDF()
    varfinal['varCode']=varfinal['name'] + varfinal['transformed'] + varfinal['source']

    datasetNames = ['previous_application.csv', 'POS_CASH_balance.csv', 'credit_card_balance.csv', 'bureau.csv']
    for dataset in datasetNames:
        print(dataset)
        data = pd.read_csv(sLoc + dataset)
        relevantIndex = data[varMan.pk]
        varfinal1 = varfinal[varfinal['source'] == dataset.split(".csv")[0]]
        relevantVars = list(varfinal1['varCode']) + [varMan.pk]
        dfinal = pd.read_csv("./rest.csv", usecols=relevantVars)
        dfinal = dfinal[dfinal[varMan.pk].isin(relevantIndex)]
        dfinal = dfinal.set_index(dataMan.pk).join(dataMan.dataCards[2].df.set_index(dataMan.pk))
        a = IV(getWoe=1, verbose=1)
        binned = a.binning(dfinal, 'TARGET', maxobjectFeatures=300, varCatConvert=1)
        ivData = a.iv_all(binned, 'TARGET')
        converted_train = a.convertToWoe(dfinal, binningOnly=0)
        output=output.join(converted_train)
        if dataset=='previous_application.csv':a.saveVarcards(dLoc, 'restWoeCards')
        else:
            a.saveVarcards(dLoc, 'temp')
            temp=pd.read_csv(dLoc+"temp.csv")
            temp.to_csv(dLoc+'restWoeCards.csv',mode='a',header=False,index= False)
    output.to_csv(dLoc+'restWoe.csv')
elif stage==161:
    dLoc = '/home/pooja/PycharmProjects/homeCredit/dataManagerFiles/train/cleaned/forModelRun/'
    output=pd.read_csv(dLoc + 'restWoe.csv')
    f=distReports(output)
    f.to_csv(dLoc+"restDescribe.csv")


elif stage==19:#get the plots
    train=pd.read_csv("/home/pooja/PycharmProjects/homeCredit/dataManagerFiles/train/cleaned/forModelRun/rest.csv")
    plotGrabh(train,'TARGET','/home/pooja/PycharmProjects/homeCredit/dataManagerFiles/train/cleaned/forModelRun/graphs/')
elif stage ==20:#make a prediction
   pass



