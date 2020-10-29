import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
validation =0
case=2
pk='PassengerId'
baseLoc='/home/pooja/PycharmProjects/titanic/'

target='Survived'

main=pd.read_csv(baseLoc+'dataManagerFiles/train/'+"mainwithCovars.csv",index_col=pk)
tar= pd.read_csv(baseLoc + 'dataManagerFiles/train/' + "target.csv", index_col=pk)
main=main.join(tar)
#testSample = pd.read_csv(baseLoc + 'dataManagerFiles/train/' + "testwithCovars.csv",index_col=pk)
#testSample.columns=[name.replace("testWithDummies","mainWithDummies") for name in testSample.columns]
test=main.sample(n=int(main.shape[0]*0.20) ,random_state=0)
#print(test.index)
train=main.drop(test.index,axis=0)
test_y=test[target]
train_y=train[target]

extraExcludes=[]#extraExcludes1+extraExcludes2

train=train.drop([target]+extraExcludes,axis=1)
test=test.drop([target]+extraExcludes,axis=1)
varSelected=['TicketMeanT0mainWithDummies','Sex0mainWithDummies']
X=train[varSelected].fillna(0)
y=train_y




clf = DecisionTreeClassifier(max_leaf_nodes=1000, random_state=0,max_depth=100)

# Train Decision Tree Classifer
clf = clf.fit(X,y)

#Predict the response for test dataset
y_pred = clf.predict(X)
test_pred=clf.predict(test[varSelected].fillna(0))

train['predicted']=y_pred
test['predicted']=test_pred
score_test = metrics.roc_auc_score(test_pred , test_y)
score_train = metrics.accuracy_score(y,train['predicted'])
print(score_train,score_test )

from sklearn import tree
import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(clf,
               feature_names = varSelected,
               class_names='TARGET',
               filled = True);
fig.savefig(baseLoc+'imagename.png')
