from abc import ABC, abstractmethod
from itertools import combinations
from dataManager import  dataObject
from commonFuncs import objectTodf,dfToObject
from dataExploration import distReports
import numpy as np
import pandas as pd
from iv import IV
import time
from multiprocessing import Pool, Process, cpu_count
class varInterface(ABC):
    def __init__(self,name,id,dataset,pk,rk=None,type=None,transformed=0):

        self.name=name
        self.id=id
        self.source=dataset
        self.pk=pk
        self.rk=rk
        self.type=type
        self.transformed=transformed

class rawVar(varInterface):

    pass

class compositeVar(varInterface):
    def __init__(self,varID,pk,vars=None,operator=None,aggregator=None):
        self.vars = vars
        self.operator = operator
        self.aggregator = aggregator
        self.varName = ["_".join(v) for v in vars.name][0]+"_"+operator
        self.varID=varID
        self.pk=pk
        self.source=set([v for v in vars.source])

class varStore():


    def __init__(self,pk):


        self.__varCount=1
        self.pk=pk
        self.funcdict={'multVar':self.multVar}




    def getVars(self,d):
        varList=self.getVarTypes(d)
        varObjects=[]
        type=['cont','cat']

        for i in [0,1]:

            for v in varList[i]:
                varObjects.append(rawVar(name=v, id=self.__varCount, dataset=d.name, pk=d.pk, rk=d.rk,type=type[i]))
                self.__varCount+=1
        return varObjects
    def multVar(self,varObjects,codes):#creates mult var
        #type condition
        varObjectsfeatured=[]
        transCodes=str(codes).split("|")


        volunteer = varObjects[0]
        source = volunteer.source
        combs = list(combinations(varObjects, 2))
        for c in combs:
            for code in transCodes:
                for t in code.split(";"):
                        m =t.split(":")
                        if m[0] in ["add", "mult", "div"]:
                            if c[0].type=='cont' and c[1].type=='cont':
                                type='cont'
                                varObjectsfeatured.append(
                                rawVar(name=c[0].name + "|" + c[1].name, id=self.__varCount, dataset=source,
                                           pk=volunteer.pk, rk=volunteer.rk, transformed=m[0], type=type))
                                self.__varCount += 1
                                if m[1]==2:
                                    rawVar(name=c[1].name + "|" + c[0].name, id=self.__varCount, dataset=source,
                                           pk=volunteer.pk, rk=volunteer.rk, transformed=m[0], type=type)
                                    self.__varCount += 1

                        elif m[0] in ['catBin']:
                            type='cat'
                            varObjectsfeatured.append(
                                rawVar(name=c[0].name + "|" + c[1].name, id=self.__varCount, dataset=source,
                                       pk=volunteer.pk, rk=volunteer.rk, transformed=m[0], type=type))
                            self.__varCount += 1

                        else:pass

        return varObjectsfeatured
    def featEng(self,d,code):
        func=self.funcdict[code]
        return func(d)
    def getVarTypes(self,d):
        d.load()
        df=d.df
        df=df.replace(np.inf,np.nan)
        objectCols = list(df.select_dtypes(include=['object']).columns)
        allCols = df.columns
        numCols = set(allCols) - set(objectCols)
        uniques = pd.DataFrame({'nuniques': df[numCols].nunique()}, index=df[numCols].columns.values)
        numCats = list(uniques[uniques['nuniques'] < 25].index)
        catCols = list(set(objectCols + numCats) -set([self.pk]))
        contCols = list(set(allCols) - set(catCols)-set([self.pk]))
        return [contCols ,catCols]

def getDiagReport(df, col=None,code=None):

        final = distReports(df)

        a = IV()
        binned = a.binning(df, col, maxobjectFeatures=300, varCatConvert=1)
        result=a.iv_all(binned, col)
        final1=result.groupby('variable')['IV'].sum()
        final1=pd.DataFrame(final1)
        final = final.join(final1)

        final['code']=code
        #print(final)
        final.to_csv('./report.csv',mode = 'a', header = False)
        df=0

class varFactory():
    def __init__(self,varList,dataCards,diag=1,target=None,targetCol=None,pk=None,batchSize=10):
        self.varDF=varList
        self.dataCards=dataCards
        self.func={'catBin':self.factory,'0':self.doNothing}
        self.diag=diag
        self.IVMan = IV(0)
        self.batchSize=batchSize
        self.pk=pk
        self.target = target.set_index(self.pk)
        self.targetCol = targetCol
        self.startTime = time.time()
        self.IVreport=pd.DataFrame


    def break1(self,source):
        df=self.varDF[self.varDF['source']==source]
        if df.shape[0]==0:return None
        new = df["name"].str.split("|", n=1, expand=True)
        # making separate first name column from new data frame
        df["var1"] = new[0]
        # making separate last name column from new data frame
        try:
            df["var2"] = new[1]
        except:
            df["var2"] = [None for i in range(len(new[0]))]
        transformation=list(df['transformed'].unique())
        var1=list(df["var1"].unique())
        dict={}

        for t in transformation:
            dict1={}
            df1 = df[df['transformed'] == t]
            if df1['var2'].values[0]==None:dict[t]=list(df1['var1'])
            else:
                for v in var1:
                    dict1[v]=list(df1[df1['var1']==v]['var2'])
                dict[t]=dict1

        return dict
    def doNothing(self,t):
        return t


    def factory(self,df1,codes,var1,var2=None):
        batch=self.batchSize

        cores = cpu_count()
        pool = Pool(processes=cores)

        code=str(codes).split(";")
        firstTime=True
        if self.targetCol in var1:return None
        elif var2 is not None:vars=var2
        else:vars=var1
        total_var = len(list(vars))
        numberOfBatches = int(total_var/ batch)+1


        if self.diag!=1:
            numberOfBatches=1
            batch=np.inf
        for i in range(0, numberOfBatches):
            varCurtailed = vars[i * batch:min(i * batch + batch,total_var)]
            for c in code:
                t=c.split(';')

                if t[0]=='catBin':

                        temp= self.IVMan.binning(df1[varCurtailed+[var1]], maxobjectFeatures=100, varCatConvert=1)
                        temp=temp.astype(str)
                        for v in varCurtailed:
                            temp[v+"|"+var1]=temp[var1]+ "_&_" + temp[v]
                        temp.drop([var1]+varCurtailed,axis=1)



                elif t[0]=='m':
                    temp = df1[varCurtailed].multiply(df1[var1])
                    temp.columns = [str(var1 + "|").join(col) for col in varCurtailed]
                elif t[0]=='div':
                    temp = df1[varCurtailed].div(df1[var1], axis = 0)
                    temp.columns = [col+"|" +var1 for col in varCurtailed]
                    v=0
                elif t[0]=='0':
                    temp=df1[varCurtailed]

                elif t[0]=='rollUp':pass
                elif t[0]=='slice':pass
                else :temp=df1

            if self.diag==1:
                if firstTime:

                    temp=temp.join( self.target[self.targetCol])
                    tar=temp[self.targetCol]
                    firstTime=False
                else:
                    temp['TARGET']=tar


                #self.getDiagReport(temp)
                pool.apply_async(getDiagReport, args=(temp,self.targetCol,codes))
            if i %  batch == 0:
                pool.close()
                pool.join()
                pool = Pool(processes=cores)


        else:
            return temp

    def produceVar(self):
        final=pd.DataFrame(columns=['varName','missing','missing_percent','count','mean','std','min','25%','50%','75%','max','IV','code'])
        final.to_csv('./report.csv',mode = 'w', header = True)
        final=pd.DataFrame()
        for d1 in self.dataCards:
            if d1.include==1:
                dict = self.break1(d1.name)
                if dict is None: continue
                d1.load()


                df=d1.df
                #if self.targetCol in df.columns: df = df.drop(self.targetCol, axis=1)
                df=df.set_index(self.pk)
                for t in dict.keys():
                    print(t)
                    if type(dict[t])!=list:
                        for v in dict[t].keys():
                            temp=self.factory(df[[v]+dict[t][v]],t,v, dict[t][v])
                    else:
                        temp=self.factory(df[dict[t]],t,dict[t])
                    final=final.append(temp)


        print(time.time()-self.startTime)
        if self.diag==1: self.IVreport
        else :return final






    def divFact(self):pass
    def multFact(self):pass
    def diagnostics(self):pass


class varOwner():
    def __init__(self,loc,pk):
        self.loc = loc
        #varCards=[]
        # except FileNotFoundError:
        self.varCards=[]
        self.pk=pk
        self.top=[]
        self.topCount=250
        self.varStore=varStore(pk=self.pk)

    def addTomainSheet(self):pass
    def start(self,dataObjectList):
        pass
    def getVarDF(self):
        temp = dataObject(loc=self.loc, name="book")
        temp.load()
        return temp.df
    def load(self):
        temp = dataObject(loc=self.loc, name="book")
        temp.load()
        self.varCards=dfToObject(temp.df, varInterface)

        self.varCards=dfToObject(temp.df,rawVar)
    def addVarFromDataObjects(self,dataObjectList):
        for d in dataObjectList:
            if d.include==1:
                rawVars=self.varStore.getVars(d)
                self.varCards.extend(rawVars)
    def addCoVar(self,dataObjectList):
        for d in dataObjectList:
            if d.include==1:
                rawVars = self.varStore.getVars(d)
                self.varCards.extend(self.varStore.multVar(rawVars,d.transformation))

    def saveVarcards(self):
        frame=objectTodf(self.varCards)
        dataList=dataObject(df=frame,name='book')
        dataList.save(self.loc)














