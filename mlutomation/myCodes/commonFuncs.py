import pandas as pd
from collections import namedtuple

def objectTodf( objectList):
    lst = []
    for v in objectList:
        lst.append(v.__dict__)
    final=pd.DataFrame(lst)
    return final
def dfToObject(df,class1):
    cols=df.columns
    return [class1(*[row[x] for x in cols ]) for  index,row in df.iterrows()]
