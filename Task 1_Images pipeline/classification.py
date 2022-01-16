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


class Classifier():

    def __init__(self, x, y, data, cluster_model, num_clust, augment=False, debug=False):

        self.x = x
        self.y = y
        self.data = data
        self.augment = augment
        self.debug = debug
        self.cluster_model = cluster_model
        self.num_clust = num_clust  # Number of clusters in the cluster model

        self.all_x = x
        self.all_y = y

        if augment:
            self.aug_x, self.aug_y = self.augment_data()
            self.all_x = self.all_x + self.aug_x
            self.all_y = self.all_y + self.aug_y

    def augment_data(self):
        """Augment the data

        Returns:
            augmented_data: The augmented data
        """
        aug_x = []
        aug_y = []
        # Augment the data
        for i in range(len(self.data['image'])):
            # Mirror the image
            mirrored_image = np.fliplr(self.data['image'][i])
            # Get the SIFT features
            sift = cv2.SIFT_create()
            img_kp, img_des = sift.detectAndCompute(
                mirrored_image, None)
            # Get the clusters
            predict_kmeans = self.cluster_model.predict(img_des)
            mirrored_hist, bin_edges = np.histogram(
                predict_kmeans, bins=self.num_clust)
            # Normalize the histogram
            mirrored_hist = mirrored_hist / len(img_kp)
            aug_x.append(mirrored_hist)
            aug_y.append(self.data['label'][i])
        return aug_x, aug_y

    def get_accuracy_cross_validation(self, model):
        # Cross validate the model
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=43)
        accuracy = cross_val_score(
            model, self.all_x, self.all_y, cv=cv)
        mean_accuracy = np.mean(accuracy)
        return mean_accuracy

    def train_KNN_model(self):
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
        knn_model = self.get_best_model(
            knn_model, knn_params, 'KNN')

        return knn_model

    def train_RF_model(self):
        """Train a random forest classifier

        Args:
            train_featvec (List): List of feature vectors
            target (List):

        Returns:
            model: trained RF model
        """

        rf = RandomForestClassifier(n_estimators=50, max_depth=3)
        rf.fit(self.all_x, self.all_y)

        return rf

    def train_SVM_model(self):
        """Train a support vector machine

        Args:
            train_featvec (List): List of feature vectors
            target (List): List of target values

        Returns:
            model: Trained SVM model
        """
        svm = SVC()

        # Train model
        svm.fit(self.all_x, self.all_y)

        return svm

    def get_best_model(self, model, params, name):
        """Get the best model for a given model and parameters

        Args:
            model (Model): Model to be trained
            params (Dict): Parameters to be used for training
            name (String): Name of the model

        Returns:
            model: Trained model
        """
        train_idxs, test_idxs = self.get_cv_split()
        # TODO add scoring (e.g. scoring = roc_auc_score)
        # n_jobs=-1 means that all CPUs will be used
        grid = GridSearchCV(model, param_grid=params,
                            cv=zip(train_idxs, test_idxs), n_jobs=-1, error_score=0.0, verbose=0)
        grid.fit(self.all_x, self.all_y)

        return grid

    def get_cv_split(self):
        strat_split = StratifiedShuffleSplit(
            n_splits=5, test_size=0.2, random_state=42)
        train_idxs = []
        test_idxs = []
        idx = 0
        for train_idx, test_idx in strat_split.split(self.x, self.y):
            train_idxs.append(train_idx)
            test_idxs.append(test_idx)  
            if self.augment:
                train_idxs[idx] = np.concatenate([train_idxs[idx],(train_idx + len(self.x))])
                test_idxs[idx] = np.concatenate([test_idxs[idx],(test_idx + len(self.y))])
            idx += 1
            
        return train_idxs, test_idxs

    def train_ensemble(self):
        """Train an ensemble of classifiers

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

        svc = self.get_best_model(clf1, svm_params, 'SVM')
        estimators.append(('SVM', svc.best_estimator_,
                           svc.best_params_, svc.best_score_))

        # Logistic Regression
        clf2 = LogisticRegression()
        logr_params = {}
        logr_params['penalty'] = ['l1', 'l2']
        logr_params['tol'] = np.logspace(-4, -1, 4)
        logr_params['C'] = np.logspace(-2, 3, 6)

        logreg = self.get_best_model(clf2, logr_params, 'Logistic Regression')
        estimators.append(('Logistic Regression', logreg.best_estimator_,
                           logreg.best_params_, logreg.best_score_))

        # Random Forest
        clf3 = RandomForestClassifier()
        rf_params = {}
        rf_params['n_estimators'] = [10, 50, 100]
        rf_params['max_depth'] = [3, 4, 5, 6, 7]

        rf = self.get_best_model(clf3, rf_params, 'Random Forest')
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
        train_idxs, test_idxs = self.get_cv_split()
        score = cross_val_score(ensemble, self.all_x, self.all_y, cv=zip(train_idxs, test_idxs))
        mean_score = np.mean(score)
        print('Ensemble scores: ', score)
        print('Ensemble mean score: ', mean_score)

        return mean_score, ensemble
