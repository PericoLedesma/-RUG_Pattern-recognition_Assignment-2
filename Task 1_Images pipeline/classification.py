from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

def do_svm(train_featvec, target):
    import pdb; pdb.set_trace()
    # Train an SVM using a grid search
    # https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
    C_range = np.logspace(-2, 10, 13)
    gammas = ['scale', 'auto']
    kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
    param_grid = dict(gamma=gammas, C=C_range, kernel=kernels)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
    grid.fit(train_featvec, target)

    print(
        "The best parameters are %s with a score of %0.2f"
        % (grid.best_params_, grid.best_score_)
    )

    return grid.best_score_

    # clf = svm.SVC(kernel='linear', C=1)
    # clf = svm.SVC()
    # clf.fit(train_featvec, target)

    # predict = clf.predict(train_featvec)

    # score = accuracy_score(np.asarray(target), predict)

    # return score
