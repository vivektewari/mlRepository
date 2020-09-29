import os
from dataManager import dataOwner,dataObject
from varManager  import varOwner
import pandas as pd
class projectOwner():
    def __init__(self,loc):
        self.loc=loc
        self.desk=loc+'projectMangerFiles/'
        self.getTaskList()
    def createFolder(self,folder,extraLoc=""):
        if not os.path.exists(self.loc+extraLoc+'/'+folder): os.mkdir(self.loc+extraLoc+'/'+folder)
    def initializeFolders(self):
        for f in ['projectMangerFiles','varManagerFiles','dataManagerFiles']:
            self.createFolder(f)
        for f in ['train','valid','test']:
            self.createFolder(f,extraLoc='dataManagerFiles/')
    def initializeEmployees(self):
        a,b=dataOwner(),varOwner()
        return a,b
    def getTaskList(self):
            self.taskFile = dataObject(loc=self.desk, name='pmFile')
            try:
             self.taskFile.load()

            except FileNotFoundError:
                self.initializeFolders()
                self.taskFile.df=pd.DataFrame({'taskNO.':[1,2,3],'task':['createDMFile','createVMFile','dummy'],'completed':[0,0,0]})
                self.taskFile.save()
    def taskMapping(self):pass

