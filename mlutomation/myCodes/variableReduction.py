import pandas as pd
a=pd.read_csv("./all_iv.csv")
b=a[a['IV']>0.03]
c=b.sort_values(ascending=False,by='IV')
d=c.drop_duplicates(keep="first",subset=['varName'])
d.to_csv("./fileterIv.csv",index=False)