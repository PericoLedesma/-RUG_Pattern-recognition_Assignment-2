import numpy as np
import pandas as pd
import cv2

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
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
    return mean_accuracy


def train_KNN_model(train_featvec, target, cluster_model, data, augment=False):
    """Train a k-nearest neighbor classifier

    Args:
        train_featvec (List): List of feature vectors
        target (List): List of target values

    Returns:
        model: Trained KNN model
    """

    knn_model = KNeighborsClassifier(n_neighbors=3)

    knn_params = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
              'weights': ['uniform', 'distance'],
              'algorithm': ['ball_tree', 'kd_tree', 'brute']}
    knn_model = get_best_model(
        knn_model, knn_params, train_featvec, target, 'KNN', cluster_model=cluster_model, data=data, augment=augment)

    return knn_model


def train_RF_model(train_featvec, target, cluster_model, data, augment=False):
    """Train a random forest classifier

    Args:
        train_featvec (List): List of feature vectors
        target (List):

    Returns:
        model: trained RF model
    """

    rf = RandomForestClassifier(n_estimators=50, max_depth=3)
    rf.fit(train_featvec, target)

    return rf


def train_SVM_model(train_featvec, target, cluster_model, data, augment=False):
    """Train a support vector machine

    Args:
        train_featvec (List): List of feature vectors
        target (List): List of target values

    Returns:
        model: Trained SVM model
    """
    svm = SVC()

    # Train model
    svm.fit(train_featvec, target)

    return svm


def get_best_model(model, params, train_featvec, target, name, cluster_model, data, augment=False):

    train_idxs, test_idxs, x, y = get_cv_split(
        train_featvec, target, cluster_model=cluster_model, data=data, augment=augment)
    # TODO add scoring (e.g. scoring = roc_auc_score)
    # n_jobs=-1 means that all CPUs will be used
    grid = GridSearchCV(model, param_grid=params,
                        cv=zip(train_idxs, test_idxs), n_jobs=-1, error_score=0.0, verbose=0)
    grid.fit(x, y)

    return grid

def get_cv_split(x, y, cluster_model, data, augment=False):
    strat_split = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    train_idxs = []
    test_idxs = []
    for train_idx, test_idx in strat_split.split(x, y):
        train_idxs.append(train_idx)
        test_idxs.append(test_idx)
    if augment:
        len_x = len(x)
        len_y = len(y)
        assert len_x == len_y
        for i in range(len(train_idxs)):
            # Get the corresponding training data
            train_x = [x[i] for i in train_idxs[i]]
            train_y = [x[i] for i in train_idxs[i]]

            print('Train shape:', str(np.shape(train_x)))
            print('Train shape:', str(np.shape(train_y)))
            # Augment the train data and add the data to x
            for img_idx, img in enumerate(train_x):
                # print(data['image'][img_idx])
                # Mirror the image
                mirrored_image = np.fliplr(data['image'][img_idx])
                # Get the SIFT features
                sift = cv2.SIFT_create()
                img_kp, img_des = sift.detectAndCompute(mirrored_image, None)
                # Get the clusters
                predict_kmeans = cluster_model.predict(img_des)
                hist, bin_edges = np.histogram(predict_kmeans, bins=20)
                # Normalize the histogram
                hist = hist / len(img_kp)

                train_idxs.append(len_x + img_idx)
                x.append(hist)
                y.append(train_y[img_idx])
    return train_idxs, test_idxs, x, y


def train_model(train_featvec, target, cluster_model, data, n_models=3, DEBUG=False, augment=False):
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

    estimators = []

    # TODO consider other parameter settings

    # Set the parameters for the grid search
    # SVM
    clf1 = SVC()
    svm_params = {}
    svm_params['C'] = np.logspace(-2, 3, 6)
    svm_params['gamma'] = ['scale', 'auto']
    svm_params['kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']

    svc = get_best_model(clf1, svm_params, train_featvec,
                         target, 'SVM', cluster_model=cluster_model, data=data, augment=augment)
    estimators.append(('SVM', svc.best_estimator_,
                      svc.best_params_, svc.best_score_))

    # Logistic Regression
    clf2 = LogisticRegression()
    logr_params = {}
    logr_params['penalty'] = ['l1', 'l2']
    logr_params['tol'] = np.logspace(-4, -1, 4)
    logr_params['C'] = np.logspace(-2, 3, 6)

    logreg = get_best_model(
        clf2, logr_params, train_featvec, target, 'Logistic Regression', cluster_model=cluster_model, data=data, augment=augment)
    estimators.append(('Logistic Regression', logreg.best_estimator_,
                      logreg.best_params_, logreg.best_score_))


    # Random Forest
    clf3 = RandomForestClassifier()
    rf_params = {}
    rf_params['n_estimators'] = [10, 50, 100]
    rf_params['max_depth'] = [3, 4, 5, 6, 7]

    rf = get_best_model(clf3, rf_params, train_featvec, target,
                        'Random Forest', cluster_model=cluster_model, data=data, augment=augment)
    estimators.append(('Random Forest', rf.best_estimator_,
                        rf.best_params_, rf.best_score_))    

    for estimator in estimators:
        print(
            "The best parameters for %s are %s with a score of %0.2f"
            % (estimator[0], estimator[2], estimator[3])
        )
    # cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

    # # TODO add scoring (e.g. scoring = roc_auc_score)
    # # n_jobs=-1 means that all CPUs will be used
    # grid = GridSearchCV(pipeline, param_grid=grid_search_params,
    #                     cv=cv, n_jobs=-1, error_score=0.0, verbose=0)
    # grid.fit(train_featvec, target)

    # # FIXME There exist many illegal comibnation os parameters
    # # This results in many warnings

    # print(
    #     "The best parameters are %s with a score of %0.2f"
    #     % (grid.best_params_, grid.best_score_)
    # )

    # # Order the results of the grid search by the best score
    # order_by_rank = pd.DataFrame(
    #     grid.cv_results_).sort_values(by='rank_test_score')

    # if DEBUG:
    #     print(order_by_rank['params'])

    # Get the top N estimators
    # for rank in range(n_models):
    #     params = order_by_rank['params'].iloc[rank]
    #     estimators.append((str(rank),
    #                        params['classifier'])
    #     )

    # Create an ensemble
    ensemble = VotingClassifier(
        estimators=[(i[0], i[1]) for i in estimators],
        voting='hard',
    )

    # Cross validate the ensemble
    train_idxs, test_idxs, x, y = get_cv_split(
        train_featvec, target, data=data, augment=augment)
    score = cross_val_score(ensemble, x, y, cv=zip(train_idxs, test_idxs))
    mean_score = np.mean(score)
    print('Ensemble scores: ', score)
    print('Ensemble mean score: ', mean_score)

    return mean_score, ensemble
