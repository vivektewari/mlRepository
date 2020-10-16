import pandas as pd
from itertools import combinations
from iv import IV
from dataExploration import distReports
dLoc='/home/pooja/PycharmProjects/homeCredit/dataManagerFiles/'
mainDatsetName='forModelRun/rest.csv'
folder="test/cleaned/"
pk='SK_ID_CURR'
case= 'posBurSegments'   #'posBurSegments' #'restSegments' #'ccSegments'
train=False
if case=='DatasetIntersection':
    main=pd.read_csv(dLoc+folder+mainDatsetName,usecols=[pk])
    main['main']=1
    main.index=main[pk]
    main=main[['main',pk]]
    datasetNames = ['previous_application.csv', 'POS_CASH_balance.csv', 'credit_card_balance.csv', 'bureau.csv']
    dict={}
    for j in datasetNames:
        df=pd.read_csv(dLoc+folder+j,usecols=[pk])
        dict[j]=df[pk]
    combs=[]
    for i in range(1,5):
        combs.append(list(combinations(datasetNames, i)))

    for comb in combs:
        for c in comb:
            name="|"

            intersection = set(main[pk])
            for c1 in c:
                name=name+c1+"|"
                intersection=set(intersection).intersection(dict[c1])
            main[name]=0
            main.loc[list(intersection), [name]]=1
    main.to_csv(dLoc+folder+"forModelRun/dataIntersection.csv")
    main.describe().to_csv(dLoc+folder+"forModelRun/dataIntersectionDescribe.csv")


if case=='ccSegments':
    tar = pd.read_csv("/home/pooja/PycharmProjects/homeCredit/tests2/dataManagerFiles/train/target.csv")
    main=pd.read_csv(dLoc + folder + "forModelRun/dataIntersection.csv")
    main=main[main['|credit_card_balance.csv|']==1]
    rest = pd.read_csv(dLoc + folder + mainDatsetName)
    rest=rest[rest[pk].isin(main[pk])]
    rest=rest.set_index(pk).join(tar.set_index(pk))
    if train:
        a = IV(getWoe=1, verbose=1)
        binned = a.binning(rest, 'TARGET', maxobjectFeatures=300, varCatConvert=1)
        ivData = a.iv_all(binned, 'TARGET')
        converted_train = a.convertToWoe(rest, binningOnly=0)
        a.saveVarcards(dLoc + folder+"forModelRun/", case + 'Cards')
    new = IV()
    new.load(dLoc + "train/cleaned/"+"forModelRun/", case + 'Cards')
    converted_train = new.convertToWoe(rest, binningOnly=0)
    converted_train.to_csv(dLoc + folder + "forModelRun/CCSegment.csv")
    a=distReports(converted_train)
    a.to_csv(dLoc + folder + "forModelRun/CCSegmentDescibe.csv")
if case=='posBurSegments':
    tar = pd.read_csv("/home/pooja/PycharmProjects/homeCredit/tests2/dataManagerFiles/train/target.csv")
    main=pd.read_csv(dLoc + folder+ "forModelRun/dataIntersection.csv")
    includeIndex=main[main['|previous_application.csv|POS_CASH_balance.csv|bureau.csv|']==1]
    excludeIndex=main[main['|credit_card_balance.csv|']==1]
    keepIndex=set(includeIndex[pk]).difference(set(excludeIndex[pk]))
    rest = pd.read_csv(dLoc + folder + mainDatsetName)
    rest=rest[rest[pk].isin(list(keepIndex))]
    rest=rest.set_index(pk).join(tar.set_index(pk))
    varExclude=[var  for var in rest.columns if 'credit_card_balance' in var ]
    rest=rest.drop(varExclude,axis=1)
    if train:
        a = IV(getWoe=1, verbose=1)
        binned = a.binning(rest, 'TARGET', maxobjectFeatures=300, varCatConvert=1)
        ivData = a.iv_all(binned, 'TARGET')
        converted_train = a.convertToWoe(rest, binningOnly=0)
        a.saveVarcards(dLoc + folder+"forModelRun/", case + 'Cards')
    new = IV()
    new.load(dLoc + "train/cleaned/" +"forModelRun/",case+'Cards')
    converted_train = new.convertToWoe(rest, binningOnly=0)
    converted_train.to_csv(dLoc + folder + "forModelRun/posBurSegments.csv")
    a=distReports(converted_train)
    a.to_csv(dLoc + folder + "forModelRun/posBurSegmentsDescibe.csv")
if case=='restSegments':
    tar = pd.read_csv("/home/pooja/PycharmProjects/homeCredit/tests2/dataManagerFiles/train/target.csv")
    main=pd.read_csv(dLoc + folder+ "forModelRun/dataIntersection.csv")
    ccIndex = main[main['|previous_application.csv|POS_CASH_balance.csv|bureau.csv|'] == 1]
    otherIndex = main[main['|credit_card_balance.csv|'] == 1]
    excludeIndex = list(set(ccIndex[pk]).union(set(otherIndex[pk])))
    keepIndex = set(main[pk]).difference(set(excludeIndex))
    rest = pd.read_csv(dLoc + folder + mainDatsetName)
    rest=rest[rest[pk].isin(list(keepIndex))]
    rest=rest.set_index(pk).join(tar.set_index(pk))
    varExclude=[var  for var in rest.columns if 'credit_card_balance' in var ]
    rest=rest.drop(varExclude,axis=1)
    if train:
        a = IV(getWoe=1, verbose=1)
        binned = a.binning(rest, 'TARGET', maxobjectFeatures=300, varCatConvert=1)
        ivData = a.iv_all(binned, 'TARGET')
        converted_train = a.convertToWoe(rest, binningOnly=0)
        a.saveVarcards(dLoc + folder+"forModelRun/", case + 'Cards')
    new = IV()
    new.load(dLoc + "train/cleaned/"+"forModelRun/", case + 'Cards')
    converted_train = new.convertToWoe(rest, binningOnly=0)
    converted_train.to_csv(dLoc + folder + "forModelRun/restSegments.csv")
    a=distReports(converted_train)
    a.to_csv(dLoc + folder + "forModelRun/restSegmentsDescibe.csv")






