import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

# https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
# https://towardsdatascience.com/how-to-tune-multiple-ml-models-with-gridsearchcv-at-once-9fcebfcc6c23

def get_accuracy_cross_validation(model, train_featvec, target):
    # Cross validate the ensemble
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=43)
    accuracy = cross_val_score(model, train_featvec, target, cv=cv)
    mean_accuracy = np.mean(accuracy)
    print('Ensemble mean score: ', mean_accuracy)
    return mean_accuracy

def train_SVM_model(train_featvec, target):
    """Train a support vector machine

    Args:
        train_featvec (List): List of feature vectors
        target (List): List of target values

    Returns:
        model: Trained SVM model
    """
    svm = SVC()

    #Train model
    svm.fit(train_featvec, target)

    return svm


def train_model(train_featvec, target, n_models = 3, DEBUG=False):
    """Train an ensemble of classifiers

    Args:
        train_featvec (List): List of feature vectors
        target (List): List of target values
        n_models (int, optional): Number of models in the ensemble. Defaults to 3.
        DEBUG (bool, optional): Debug mode. Defaults to False.

    Returns:
        tuple: (mean score, ensemble model)
    """

    #import pdb; pdb.set_trace()

    # Set the parameters for the grid search
    # SVM
    clf1 = SVC()
    # TODO extend the amount of parameters
    svm_params = {}
    svm_params['classifier__C'] = np.logspace(-2, 3, 6)
    svm_params['classifier__gamma'] = ['scale', 'auto']
    svm_params['classifier__kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']
    svm_params['classifier'] = [clf1]

    # Logistic Regression
    clf2 = LogisticRegression()
    # TODO extend the amount of parameters
    logr_params = {}
    logr_params['classifier__penalty'] = ['l1', 'l2']
    logr_params['classifier__tol'] = np.logspace(-4, -1, 4)
    logr_params['classifier__C'] = np.logspace(-2, 3, 6)
    logr_params['classifier'] = [clf2]

    clf3 = LinearRegression()
    # TODO extend the amount of parameters
    linr_params = {}
    linr_params['classifier'] = [clf3]


    # Create the pipeline
    pipeline = Pipeline([('classifier', SVC())])
    grid_search_params = [svm_params, logr_params, linr_params]


    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

    # TODO add scoring (e.g. scoring = roc_auc_score)
    # n_jobs=-1 means that all CPUs will be used
    grid = GridSearchCV(pipeline, param_grid=grid_search_params,
                        cv=cv, n_jobs=-1, error_score=0.0, verbose=0)
    grid.fit(train_featvec, target)

    # FIXME There exist many illegal comibnation os parameters
    # This results in many warnings

    print(
        "The best parameters are %s with a score of %0.2f"
        % (grid.best_params_, grid.best_score_)
    )

    # Order the results of the grid search by the best score
    order_by_rank = pd.DataFrame(
        grid.cv_results_).sort_values(by='rank_test_score')

    if DEBUG:
        print(order_by_rank['params'])

    # Get the top N estimators
    estimators = []
    for rank in range(n_models):
        params = order_by_rank['params'].iloc[rank]
        estimators.append((str(rank),
                           params['classifier'])
        )

    # Create an ensemble
    ensemble = VotingClassifier(
        estimators=estimators,
        voting='hard',
    )

    # Cross validate the ensemble
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=43)
    score = cross_val_score(ensemble, train_featvec, target, cv=cv)
    mean_score = np.mean(score)
    print('Ensemble scores: ', score)
    print('Ensemble mean score: ', mean_score)

    return mean_score, ensemble
