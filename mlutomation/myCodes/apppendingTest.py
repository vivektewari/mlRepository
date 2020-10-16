import pandas as pd

dLoc='/home/pooja/PycharmProjects/homeCredit/dataManagerFiles/test/cleaned/forModelRun/'
dataset=['posBurSegments.csv','restSegments.csv',"CCSegment.csv"]
final=pd.DataFrame()
for d in dataset:
    df=pd.read_csv(dLoc+d+"sub.csv")
    final=pd.concat([final,df],axis=0)
final.to_csv(dLoc+"sub.csv")
