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

from feature_extraction import cluster_sift_descriptions, calculate_histogram

# https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
# https://towardsdatascience.com/how-to-tune-multiple-ml-models-with-gridsearchcv-at-once-9fcebfcc6c23


class Classifier():

    def __init__(self, data, cluster_model, num_clust, augment=False, debug=False):

        self.sift_des = data['sift_description']
        self.keyp = data['keypoints']
        self.y = data['label']
        self.data = data
        self.augment = augment
        self.debug = debug
        self.cluster_model = cluster_model
        self.num_clust = num_clust  # Number of clusters in the cluster model

        self.all_sift_des = self.sift_des
        self.all_keyp = self.keyp
        self.all_y = self.y

        if augment:
            self.aug_sift_des, self.aug_keyp, self.aug_y = self.augment_data()
            # TODO check list concatenation
            self.all_sift_des = self.all_sift_des + self.aug_sift_des
            self.all_keyp = self.all_keyp + self.aug_keyp
            self.all_y = self.all_y + self.aug_y

    def augment_data(self):
        """Augment the data

        Returns:
            aug_sift_des (List): Augmented sift descriptions
            aug_y (List): Augmented target values
        """
        aug_sift_des = []
        aug_keyp = []
        aug_y = []
        # Augment the data
        for i in range(len(self.data['image'])):
            # Mirror the image
            mirrored_image = np.fliplr(self.data['image'][i])
            # Get the SIFT features
            sift = cv2.SIFT_create()
            img_kp, img_des = sift.detectAndCompute(
                mirrored_image, None)
            aug_sift_des.append(img_des)
            aug_keyp.append(img_kp)
            # # Get the clusters
            # predict_kmeans = self.cluster_model.predict(img_des)
            # mirrored_hist, bin_edges = np.histogram(
            #     predict_kmeans, bins=self.num_clust)
            # # Normalize the histogram
            # mirrored_hist = mirrored_hist / len(img_kp)
            # aug_x.append(mirrored_hist)
            # aug_y.append(self.data['label'][i])
        return aug_sift_des, aug_keyp, aug_y

    def get_accuracy_cross_validation(self, model, model_name):
        # Create cross validation splits
        self.train_idxs, self.test_idxs = self.get_cv_split()
        assert len(self.train_idxs) == len(self.test_idxs)

        # For each split do:
        accuracies = []
        for i, index_list in enumerate(range(len(self.train_idxs))):
            # Create the training set
            train_sift_des = [self.all_sift_des[k]
                              for k in index_list]
            train_keyp = [self.all_keyp[k] for k in index_list]
            train_y = [self.all_y[k] for k in index_list]

            # Create cluster model based on train set
            cluster_model = cluster_sift_descriptions(
                train_sift_des, NUM_CLUSTERS=self.num_clust)

            # Convert all the sift features to clusters in a histogram
            X, y = calculate_histogram(
                train_sift_des, train_keyp, train_y, cluster_model, self.num_clust, VISUALIZE=True)

            # Create the test set
            test_sift_des = [self.all_sift_des[k]
                             for k in self.test_idxs[i]]
            test_keyp = [self.all_keyp[k]
                         for k in self.test_idxs[i]]
            test_y = [self.all_y[k] for k in self.test_idxs[i]]

            # Convert all the sift features to clusters in a histogram
            X_test, y_test = calculate_histogram(
                test_sift_des, test_keyp, test_y, cluster_model, self.num_clust, VISUALIZE=True)

            # Train a model using the train set
            model.fit(X, y)

            # Test the model using the test set
            y_pred = model.predict(X_test)

            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            print("Accuracy: ", accuracy)

            # Store the accuracy
            accuracies.append(accuracy)
        
        print("All acc ", accuracies, ".\tMean accuracy: ", np.mean(accuracies), " for ", model_name, '')
        
        return np.mean(accuracies)


    def get_cv_split(self):
        strat_split = StratifiedShuffleSplit(
            n_splits=5, test_size=0.2, random_state=42)
        train_idxs = []
        test_idxs = []
        idx = 0
        for train_idx, test_idx in strat_split.split(self.sift_des, self.y):
            train_idxs.append(train_idx)
            test_idxs.append(test_idx)  
            if self.augment:
                train_idxs[idx] = np.concatenate([train_idxs[idx],(train_idx + len(self.sift_des))])
                test_idxs[idx] = np.concatenate([test_idxs[idx],(test_idx + len(self.y))])
            idx += 1
        return train_idxs, test_idxs

    def get_svm(self):
        svm_params = {}
        svm_params['C'] = np.logspace(-2, 3, 6)
        svm_params['gamma'] = ['scale', 'auto']
        svm_params['kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']

        mean_accuracies = []
        models = []
        for i in range(len(svm_params['C'])):
            for j in range(len(svm_params['gamma'])):
                for k in range(len(svm_params['kernel'])):
                    # Create a pipeline
                    pipe = Pipeline([('svm', SVC(C=svm_params['C'][i],
                                                 gamma=svm_params['gamma'][j],
                                                 kernel=svm_params['kernel'][k]))])
                    models.append(pipe)

                    # TODO use something else than accuracy
                    # Get the cross validated accuracy
                    mean_acc = self.get_accuracy_cross_validation(pipe, 'SVM')
                    mean_accuracies.append(mean_acc)

        # Store the best model based on the accuracy
        best_model = models[np.argmax(mean_accuracies)]
        print("Best model: ", best_model,
              " with accuracy: ", np.max(mean_accuracies))
        return best_model

    def get_logreg(self):
        logr_params = {}
        logr_params['penalty'] = ['l1', 'l2']
        logr_params['tol'] = np.logspace(-4, -1, 4)
        logr_params['C'] = np.logspace(-2, 3, 6)

        mean_accuracies = []
        models = []
        for i in range(len(logr_params['penalty'])):
            for j in range(len(logr_params['tol'])):
                for k in range(len(logr_params['C'])):
                    # Create a pipeline
                    pipe = Pipeline([('logreg', LogisticRegression(
                        penalty=logr_params['penalty'][i], tol=logr_params['tol'][j], C=logr_params['C'][k]))])
                    models.append(pipe)

                    # TODO use something else than accuracy
                    # Get the cross validated accuracy
                    mean_acc = self.get_accuracy_cross_validation(pipe, "LogReg")
                    mean_accuracies.append(mean_acc)

        # Store the best model based on the accuracy
        best_model = models[np.argmax(mean_accuracies)]
        print("Best model: ", best_model,
              " with accuracy: ", np.max(mean_accuracies))
        return best_model

    def get_rf(self):
        rf_params = {}
        rf_params['n_estimators'] = [10, 50, 100]
        rf_params['max_depth'] = [3, 4, 5, 6, 7]

        mean_accuracies = []
        models = []
        for i in range(len(rf_params['n_estimators'])):
            for j in range(len(rf_params['max_depth'])):
                # Create a pipeline
                pipe = Pipeline([('rf', RandomForestClassifier(
                    n_estimators=rf_params['n_estimators'][i], max_depth=rf_params['max_depth'][j]))])
                models.append(pipe)

                # TODO use something else than accuracy
                # Get the cross validated accuracy
                mean_acc = self.get_accuracy_cross_validation(pipe, "RF")
                mean_accuracies.append(mean_acc)
            
        # Store the best model based on the accuracy
        best_model = models[np.argmax(mean_accuracies)]
        print("Best model: ", best_model, " with accuracy: ", np.max(mean_accuracies))
        return best_model

    def get_knn(self):
        knn_model = KNeighborsClassifier()
        knn_params = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
                      'weights': ['uniform', 'distance'],
                      'algorithm': ['ball_tree', 'kd_tree', 'brute']}
        
        mean_accuracies = []
        models = []
        for i in range(len(knn_params['n_neighbors'])):
            for j in range(len(knn_params['weights'])):
                for k in range(len(knn_params['algorithm'])):
                    # Create a pipeline
                    pipe = Pipeline([('knn', KNeighborsClassifier(
                        n_neighbors=knn_params['n_neighbors'][i],
                        weights=knn_params['weights'][j],
                        algorithm=knn_params['algorithm'][k]))])
                    models.append(pipe)

                    # TODO use something else than accuracy
                    # Get the cross validated accuracy
                    mean_acc = self.get_accuracy_cross_validation(pipe, "KNN")
                    mean_accuracies.append(mean_acc)

        # Store the best model based on the accuracy
        best_model = models[np.argmax(mean_accuracies)]
        print("Best model: ", best_model, " with accuracy: ", np.max(mean_accuracies))
        return best_model

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
        svm = self.get_svm()
        estimators.append(('SVM', svm))

        # Logistic Regression
        logreg = self.get_logreg()
        estimators.append(('Logistic Regression', logreg))

        # Random Forest
        rf = self.get_rf()
        estimators.append(('Random Forest', rf))

        # for estimator in estimators:
        #     print(
        #         "The best parameters for %s are %s with a score of %0.2f"
        #         % (estimator[0], estimator[2], estimator[3])
        #     )
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
        mean_acc = self.get_accuracy_cross_validation(ensemble, "Ensemble")
        print('Ensemble scores: ', mean_acc)

        return mean_acc, ensemble
