from abc import ABC, abstractmethod
import pandas as pd
import os
from commonFuncs import objectTodf,dfToObject
from sklearn.model_selection import train_test_split
from dataExploration import distReports

class dataObject():
    def __init__(self,  name, loc=None, primaryKey=None, rollupKey=None,shape=None,match=None,include=0,transformation=0,use=None,df=None,loaded=0):
        self.name = name
        self.loc=loc
        self.pk=primaryKey
        self.rk=rollupKey
        self.shape=shape
        self.match=match
        self.include=include
        self.transformation=transformation
        self.use= use
        self.df=df
        self.loaded=loaded



    def getMetrics(self,primarykeys=None):
        if self.df==None:self.load()
        self.shape=self.df.shape
        if primarykeys is not None and self.pk is not None and self.df is not None:
            self.match=len(set(primarykeys).intersection(set(self.df)))/len(primarykeys)
    def unload(self):
        self.df=None
        self.loaded=0
    def load(self):
        if self.loaded!=1:
            loc1= self.loc + self.name + '.csv'
            self.df=pd.read_csv(loc1)
        else :pass
    def save(self, loc=None):

        if loc is None :loc1=self.loc+""
        else: loc1=loc
        loc1 = loc1 + self.name + '.csv'
        self.df.to_csv(loc1,index=False)
    classmethod
    def extractDatas(loc):
        dataCards=[]
        files = os.listdir(loc)
        for file in files:
            if ".csv" in file:
                dataCards.append(dataObject(loc=loc, name=file.replace(".csv","")))
        return dataCards


    classmethod
    def trainValidSplit( df, trainSize=0.7,loc=None):
        validSize = 1 - trainSize
        train, valid = train_test_split(df, train_size=trainSize,
                                                  test_size=validSize, random_state=0, shuffle=True)
        if loc is not None:
            train.to_csv(loc+"train.csv")
            valid.to_csv(loc+"valid.csv")
    def append(self,df):
        self.df.append(df)
    def getVariables(self):
        temp=pd.read_csv(self.loc+self.name+".csv",nrows=1)
        return list(set(temp.columns).difference(set([self.pk,self.rk])))








class dataOwner():
    def __init__(self, loc, pk=None, targetVar=None):
        self.mainKey=pk
        self.target=targetVar
        self.loc=loc
        self.pk=pk
        self.dataLoc=self.loc+"datasets/"
        self.dataCards=[]


    def load(self):
        temp = dataObject(loc=self.loc, name="book")
        temp.load()
        self.dataCards = dfToObject(temp.df, dataObject)

    def addDatacards(self,loc):
        """
        from loc picks all the dataframes and add the data owner list
        :param loc:
        :return:
        """
        self.dataCards=dataObject.extractDatas(loc)
        self.saveDatacards()
    def saveDatacards(self):
        frame=objectTodf(self.dataCards)
        dataList=dataObject(df=frame,name='book')
        dataList.save(self.loc)

    def getRelevantData(self):
            for card in self.dataCards:

                if card.use in ['train','valid','test']:
                    card.load()
                    relevantpks=list(card.df[self.pk])
                    folderName=card.use
                    for card in self.dataCards:
                        if card.use in ['train', 'valid','test']:
                            pass
                        else:
                            card.load()
                            temp=card.df
                            saveToLoc = self.loc + folderName + "/"
                            relevantDF=temp[temp[self.pk].isin(relevantpks)]
                            do=dataObject(name=card.name,df=relevantDF,primaryKey=card.pk,rollupKey=card.pk,include=card.include,loc=saveToLoc)
                            do.save()

    def getInitialReports(self):
        for card in self.dataCards:
            card.load()
            temp=distReports(card.df,detail=True)
            d=dataObject(df=temp,name=card.name+"describe",loc=card.loc,loaded=1)
            d.save()
            d.unload()
            card.unload()






















