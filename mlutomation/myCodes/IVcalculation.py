import warnings
from dataExploration import distReports,plotGrabh
from iv import IV
import pandas as pd
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
loc="/home/pooja/PycharmProjects/datanalysis/finalDatasets/"
#
a=IV(1)
#train=pd.read_csv(loc+"relevantDatasets/"+"train.csv")
#loc="/home/pooja/PycharmProjects/datanalysis/bureau/"
train= pd.read_csv('/home/pooja/PycharmProjects/datanalysis/feature_cross/crossDataset.csv')#pd.read_csv(loc+"bur.csv")
test=pd.read_csv('/home/pooja/PycharmProjects/datanalysis/rawDatas/application_test.csv')
#holdOut=train.set_index('SK_ID_CURR')
train =train.set_index('SK_ID_CURR')
test=test.set_index('SK_ID_CURR')
#train=train
#a=plotGrabh(train,'TARGET',loc+"images/")
binned=a.binning(train,'TARGET',maxobjectFeatures=300,varCatConvert=1)

#binned.to_csv('/home/pooja/PycharmProjects/datanalysis/feature_cross/crossDataset_binned.csv')
#binned=binned.join(train)
#binned=a.binning(train,'TARGET',maxobjectFeatures=300)
ivData=a.iv_all(binned,'TARGET')
converted_train=a.convertToWoe(test)
missings = converted_train[converted_train.isnull().any(axis=1)]
f=0
converted_train.to_csv("/home/pooja/PycharmProjects/datanalysis/feature_cross/test_woe.csv")
#writer = pd.ExcelWriter(loc+"iv3.xlsx")
#ivData.to_excel(writer,sheet_name="iv_detailed")
#ivData.groupby('variable')['ivValue'].sum().to_excel(writer,sheet_name="iv_summary")
#ivData.to_csv(loc+"iv_detailed_cross.csv")
#ivData.groupby('variable')['ivValue'].sum().to_csv(loc+"iv_sum_cross2.csv")

# ivInfo=pd.read_csv(loc+"iv3.csv")
# distRepo=distReports(train,ivInfo)
# distRepo.to_csv(loc+"summary.csv")
#writer.save()
#writer.close()
