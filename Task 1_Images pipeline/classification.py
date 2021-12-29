from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np

def do_svm(train_featvec, target):
    import pdb; pdb.set_trace()
    clf = svm.SVC()
    clf.fit(train_featvec, target)

    predict = clf.predict(train_featvec)

    score = accuracy_score(np.asarray(target), predict)

    return score
