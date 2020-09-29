from projectManager import  projectOwner
from dataManager import dataOwner,dataObject
from varManager import varOwner,varFactory
baseLoc='/home/pooja/PycharmProjects/homeCredit/tests2/'
stage=9

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
    factoryMan=varFactory(varMan.getVarDF(),dataMan.dataCards,diag=1,target=dataMan.dataCards[2].df,pk='SK_ID_CURR',targetCol='TARGET')
    factoryMan.produceVar()