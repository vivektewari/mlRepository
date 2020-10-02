import pandas as pd
import numpy as np

import warnings
class variable():
    def __init__(self,name,type):
        """
        :param name: string
        :param type: string|dtype cont or object
        :param type: dict|maps bin to value
        """
        self.name=name
        self.type=type
        self.bins=[]
        self.woe=[]
        self.missingWoe=0
        self.catDictionary={}
    def addMap(self,bin,value1=None):
        if bin in self.map.keys(): warnings.warn('Bin already exist. Bin value has been over ridden')
        self.map[bin]=value1
    def numConsitency(self):pass
    def applyMap(self,df,variable):
        """

        :param df: Dataframe
        :param variable: String |variable for which values are been replace
        :return: Dataframe|variable column been converted
        """
        if self.type=='cont':df[variable] = pd.cut(x=df[variable], bins=self.bins,lables=self.BinValue)
        elif self.type=='cat':df[variable]=df[variable].apply(lambda row:self.map[row])






class IV():
    def __init__(self,getWoe=0,verbose=0):
        """
        :param variables: variables Dict of variable |

        """
        if getWoe==1:
            self.variables={}
            self.getWoe=1

        else :self.getWoe=0
        self.modeBinary=1
        self.verbose=verbose
    def calculate_woe_iv(self,dataset, feature, target):
        """

        :param dataset: Dataframe with target variable
        :param feature: string|variable for which WOE needs to be calculated
        :param target: String|target Varaibe which should take value 1 or 0
        :return: Dataframe,float | Datatframe having all the partion of the variable . IV value is a number .
        """
        lst = []
        uniqueValues=list(dataset[feature].unique())
        for i in range(len(uniqueValues)):
            val = uniqueValues[i]
            lst.append({
                'Value': val,
                'All': dataset[dataset[feature] == val].count()[feature],
                'Good': dataset[(dataset[feature] == val) & (dataset[target] == 0)].count()[feature],
                'Bad': dataset[(dataset[feature] == val) & (dataset[target] == 1)].count()[feature]
            })

        dset = pd.DataFrame(lst)

        dset['Distr_Good'] = dset['Good'] / dset['Good'].sum()
        dset['Distr_Bad'] = dset['Bad'] / dset['Bad'].sum()
        dset['WoE'] = np.log(dset['Distr_Good'] / dset['Distr_Bad'])
        dset = dset.replace({'WoE': {np.inf: 0, -np.inf: 0}})
        dset['IV'] = (dset['Distr_Good'] - dset['Distr_Bad']) * dset['WoE']
        if self.modeBinary==1:iv = dset['IV'][1]
        else :iv = dset['IV'].sum()

        dset = dset.sort_values(by='WoE')
        dset1=dset
        if self.getWoe==1:
            temp=feature
            #var=temp[0]
            #value1=temp[1]
            if self.modeBinary==0:
                variable=self.variables[feature]
                # accounting missing
                if dset[dset['Value'] == 'Missing'].shape[0] > 0:
                    variable.missingWoe = list(dset[dset['Value'] == 'Missing']['WoE'])[0]
                    if variable.type=='cat':variable.bins=list(variable.bins).remove('Missing')
                else:variable.missingWoe = 0


                dset = dset[dset['Value'] != 'Missing']


                if variable.type == 'cont':

                        dset['upper']=dset['Value'].apply(lambda row:  row.right)
                        dset['lower'] = dset['Value'].apply(lambda row: row.left)
                        dset=dset.sort_values(by='upper')
                        variable.bins=[list(dset['lower'])[0]]+list(dset['upper'])
                        variable.woe = list(dset['WoE'])

                elif variable.type == 'cat':
                    bins = list(dset['Value'])
                    woe = list(dset['WoE'])
                    tempDict={}
                    for i in range(0,len(bins)):
                            tempDict[bins[i]]=woe[i]
                    variable.catDictionary=tempDict
        return dset1, iv


    def convertToWoe(self,df,target=None):
        """

        :param df:DataFrame|on which IV needs to be binning needs to be applied
        :return:DataFrame|converted to Woe values

        """

        df1=df.replace(np.inf,np.nan)
        output=df1[[]]
        if self.verbose==1:print("starting convertToWoe")
        for var in self.variables.values():
            if self.verbose==1:print(var.name)
            temp=df1[[var.name]]
            #print(var.name,var.type)
            missings = temp[temp.isnull().any(axis=1)].index # 'getting missing dataset'
            temp.loc[missings,[var.name]] =var.missingWoe
            remainingIndexes=temp.drop(missings,axis=0).index
            #temp = temp.drop(missings.index, axis=0)  # 'non missing dataset'
            if var.type == 'cat':
                indexes = []
                newCats = list(set(temp[var.name].unique()) - set(var.catDictionary.keys()))
                remainingIndexes=remainingIndexes #.to_list()
                if len(newCats) > 0:  # new category which is not in train sample
                    warnings.warn(var.name + ":New category found, assigning missing mapping")
                    indexes = []  # collecting indexes with new value
                    for element in newCats:
                        indexes = indexes+temp[temp[var.name] == element].index.to_list()
                    temp.loc[indexes,[var.name]]= var.missingWoe
                    remainingIndexes=list(temp.drop(indexes,axis=0).index)
                    #temp2 = temp.drop(indexes, axis=0)

                temp.loc[remainingIndexes,[var.name]]= temp.loc[remainingIndexes][var.name].apply(lambda row:var.catDictionary[row])



            elif var.type == 'cont':

                temp.loc[remainingIndexes,[var.name]] = pd.cut(temp.loc[remainingIndexes][var.name], bins=var.bins, labels=var.woe ,ordered=False)
                temp=temp.fillna(var.missingWoe)
            output=output.join([temp])
        if target is not None:output=output.join(df[target])
        return output

        # ivFile=pd.read_csv(iVfileLoc)
        # df=pd.read_csv(dfLoc)
        # #df=df.set_index('n_SK_ID_CURR')
        # varList=list(set((ivFile['variable'])))
        # #varList=varList.remove(nn)
        # for var in varList:
        #     #print(var)
        #     dict={}
        #     #var1=var.replace("n_","").replace('c_',"")
        #
        #     varSpecific=list(ivFile[ivFile['variable']==var]['varDetail'])
        #
        #     WoEs = list(ivFile[ivFile['variable'] == var]['WoE'])
        #     var = var.replace("j", "j_")
        #     for i in range(len(varSpecific)):
        #         if var.find("j_")>0:t1=varSpecific[i].split("_____")
        #         else : t1 = varSpecific[i].split("____")
        #
        #         dict[t1[1]]=WoEs[i]
        #
        #     try:
        #
        #         df[var]=df[var].apply(lambda row:dict[row])
        #     except:
        #         if var in df.columns: print(1)
        #         print(var)
        #     v=0
        # return df
    #x=convertToWoe("/home/pooja/PycharmProjects/datanalysis/feature_cross/crossDataset_binned.csv","/home/pooja/PycharmProjects/datanalysis/finalDatasets/iv_detailed_cross.csv" )
    #x.to_csv("/home/pooja/PycharmProjects/datanalysis/feature_cross/final_binned.csv")





    def binning(self,df, target=None, qCut=10, maxobjectFeatures=50,varCatConvert=0,excludedList=[]):
        """
        Variable binning
        :param df: Dataframe
        :param target: string
        :param qCut: int |Number of partion for the variable
        :param maxobjectFeatures: int| how many allowed feature for each character variable type. If more then variable is excluded
        :param varCatConvert: 1 or 0| if 1 returns binned variable in character format without one hot encoding. 0 uses one Hot encoding
        :param excludedList: [string]|variable which will excluded from analysis
        :return: Dataframe|Binned variable with same Shape[0] but different Shape[1] depending on varCatConvert use.
        """
        output = pd.DataFrame(index=df.index, columns=[])
        df=df.replace([np.inf,-np.inf],np.nan)
        objectCols = list(df.select_dtypes(include=['object']).columns)
        allCols = df.columns
        if target is not None: allCols = list(set(allCols) - set([target]))
        numCols = set(allCols) - set(objectCols)



        uniques = pd.DataFrame({'nuniques': df[numCols].nunique()}, index=df[numCols].columns.values)
        numCats = list(uniques[uniques['nuniques'] < 50].index)
        catCols = objectCols + numCats
        contCols = list(set(allCols) - set(catCols))
        if self.verbose == 1: print("starting binning")
        for feature in contCols:
            if self.verbose == 1: print(feature)
            temp = df[[feature]]
            #print(feature)
            missings = temp[temp.isnull().any(axis=1)]  # 'getting missing dataset'
            missings[feature] = 'Missing'

            temp = temp.drop(missings.index, axis=0)  # 'non missing dataset'

            try:

                temp[ feature],bins= pd.qcut(temp[feature], q=qCut, duplicates='drop',retbins=True)

            except IndexError:
                dices = min(df[feature].nunique(), qCut)
                if dices != 0 and temp.shape[0] > 0:

                    temp["n_" + feature],bins = pd.qcut(temp[feature], q=dices, duplicates='drop',retbins=True)
            # print(temp.shape[0])
            if self.getWoe==1:
                varObject=variable(feature,'cont')
                varObject.bins=bins
                self.variables[feature] = varObject
                varObject.woe = ["" for i in range(len(varObject.bins))]
            output = output.join(temp.append(missings))
            # print(output.shape[0])
        for feature in catCols:
            if self.verbose == 1: print(feature)
            # print(feature)
            temp = df[[feature]]
            temp = temp.fillna('Missing')
            temp[feature] = temp[feature]
            temp[feature].nunique()
            if temp[feature].nunique() > maxobjectFeatures:
                excludedList.append(feature)
                print("removed as too many categories:" + feature)
            else:
                output = output.join(temp)
            if self.getWoe==1:
                varObject = variable(feature, 'cat')
                varObject.bins = temp[feature].unique()
                varObject.woe=[None for i in range(len(varObject.bins))]

                self.variables[feature] = varObject
        if varCatConvert==1:
                self.modeBinary = 0
                if target is None: return output #.join(df[[target]])
                else: return output.join(df[[target]])

        # for col in output.columns:
        #     # X[col] = X[col].astype('category',copy=False)
        #     dummies = pd.get_dummies(output[col], prefix=col + '___')
        #     output = pd.concat([output, dummies], axis=1)
        #     output = output.drop(col, axis=1)
        # if target is not None:output = output.join(df[[target]])
        # return output


    def iv_all(self,X,target):
        """

        :param X: Dataframe| for IV calculation
        :param target: String|target variable in the dataset
        :param modeBinary: 1 or 0|
        :return:
        """
        if self.verbose == 1: print("starting IV")
        ivData=pd.DataFrame()
        #ivData=pd.DataFrame(index=X.columns,columns=['ivValue','WoE','Dist_Good','Dist_Bad','%popuation','badRate','variable'])
        for col in X.columns:
            if self.verbose == 1: print(col)
            if col == target: continue
            else:
                #print('WoE and IV for column: {}'.format(col))
                df, iv = self.calculate_woe_iv(X[[col,target]], col, target)
                df['variable']=col
                ivData=ivData.append(df)
                # if self.modeBinary ==0:
                #     ivData['ivValue'][col] = iv
                #     ivData['variable'][col] = col
                #
                # else:
                #     df.index=df.Value
                #     ivData['varDetail']=col
                #     #print(df)
                #     ivData['ivValue'][col]=df['IV'][1]
                #     ivData['Dist_Good'][col] = df['Distr_Good'][1]
                #     ivData['Dist_Bad'][col] = df['Distr_Bad'][1]
                #     ivData['WoE'][col] = df['WoE'][1]
                #     ivData['badRate'][col] = df['Bad'][1] / float(df['All'][1])
                #     ivData['%popuation'][col] = df['All'][1]/rowCount
                #     ivData['variable'][col]=col.split('___')[0]


        return ivData
if __name__ == '__main__':
    pass    #print (3 in I.open_closed(1,3))
