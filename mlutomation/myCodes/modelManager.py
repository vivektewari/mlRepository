from sklearn import metrics
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
import numpy as np
class model():
    def __init__(self,fitMethod,targetCol):
        self.rocTraining=0
        self.rocValid = 0
        self.targetCol=targetCol
    def getRoc(self,dataset,actual=None,predicted=None):
        return metrics.roc_auc_score(dataset[actual], dataset[predicted])




class modelOwner():
    def __init__(self,train,valid,test,targetCol):
        pass
    def

        clf = DecisionTreeClassifier()
        trans = RFECV(clf)
        kepler_X_trans = trans.fit_transform(kepler_X, kepler_y)
        columns_retained_RFECV = kepler.iloc[:, 1:].columns[trans.get_support()].values
    def getImportanceShuffle(self,X,Y):
        rf = RandomForestRegressor()
        scores = defaultdict(list)
        names=X.columns
        for train_idx, test_idx in ShuffleSplit(n_splits=100,train_size=0.8):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            r = rf.fit(X_train, Y_train)

            acc = r2_score(Y_test, rf.predict(X_test))
            for i in range(X.shape[1]):
                X_t = X_test.copy()
                np.random.shuffle(X_t[:, i])
                shuff_acc = r2_score(Y_test, rf.predict(X_t))
                scores[names[i]].append((acc - shuff_acc) / acc)
            print("Features sorted by their score:")
            print("sorted([(round(np.mean(score), 4), feat) for
                feat, score in scores.items()], reverse=True)
        print
        "Features sorted by their score:"
        print
        sorted([(round(np.mean(score), 4), feat) for
                feat, score in scores.items()], reverse=True)

