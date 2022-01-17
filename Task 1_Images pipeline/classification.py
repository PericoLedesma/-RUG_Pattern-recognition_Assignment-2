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

    def __init__(self, data, num_clust, augment=False, debug=False):

        self.sift_des = data['sift_description']
        self.keyp = data['sift_keypoints']
        self.y = data['label']
        self.data = data
        self.augment = augment
        self.debug = debug
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

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

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
            aug_y.append(self.data['label'][i])
            # # Get the clusters
            # predict_kmeans = self.cluster_model.predict(img_des)
            # mirrored_hist, bin_edges = np.histogram(
            #     predict_kmeans, bins=self.num_clust)
            # # Normalize the histogram
            # mirrored_hist = mirrored_hist / len(img_kp)
            # aug_x.append(mirrored_hist)
            # aug_y.append(self.data['label'][i])
        return aug_sift_des, aug_keyp, aug_y

    def get_accuracy_cross_validation(self, model, model_name, n_splits=5):
        # Create cross validation splits
        if self.X_train is None or self.y_train is None or self.X_test is None or self.y_test is None:
            self.X_train, self.y_train, self.X_test, self.y_test = self.get_cv_split(n_splits)

        # For each split do:
        accuracies = []
        for i in range(n_splits):
            # Train a model using the train set
            model.fit(self.X_train[i], self.y_train[i])

            # Test the model using the test set
            y_pred = model.predict(self.X_test[i])

            # Calculate accuracy
            accuracy = accuracy_score(self.y_test[i], y_pred)
            print("Accuracy: ", accuracy)

            # Store the accuracy
            accuracies.append(accuracy)
        
        print("All acc ", accuracies, ".\tMean accuracy: ", np.mean(accuracies), " for ", model_name, '')
        
        return np.mean(accuracies)


    def get_cv_split(self, n_splits=5):
        strat_split = StratifiedShuffleSplit(
            n_splits=n_splits, test_size=0.2, random_state=42)
        train_idxs = []
        test_idxs = []
        idx = 0
        # Create lists of indexes for train and test sets
        for train_idx, test_idx in strat_split.split(self.sift_des, self.y):
            train_idxs.append(train_idx)
            test_idxs.append(test_idx)  
            if self.augment:
                train_idxs[idx] = np.concatenate([train_idxs[idx],(train_idx + len(self.sift_des))])
                test_idxs[idx] = np.concatenate([test_idxs[idx],(test_idx + len(self.y))])
            idx += 1
        
        X_train = []
        y_train = []
        X_test = []
        y_test = []

        assert len(train_idxs) == len(test_idxs)
        for i, index_list in enumerate(train_idxs):
            # Create the training set
            train_sift_des = [self.all_sift_des[k]
                              for k in index_list]
            train_keyp = [self.all_keyp[k] for k in index_list]
            train_y = [self.all_y[k] for k in index_list]

            # Create cluster model based on train set
            cluster_model = cluster_sift_descriptions(
                train_sift_des, NUM_CLUSTERS=self.num_clust)

            # Convert all the sift features to clusters in a histogram
            X_tr, y_tr = calculate_histogram(
                train_sift_des, train_keyp, train_y, cluster_model, self.num_clust, VISUALIZE=False)

            # Create the test set
            test_sift_des = [self.all_sift_des[k]
                             for k in test_idxs[i]]
            test_keyp = [self.all_keyp[k]
                         for k in test_idxs[i]]
            test_y = [self.all_y[k] for k in test_idxs[i]]

            # Convert all the sift features to clusters in a histogram
            X_te, y_te = calculate_histogram(
                test_sift_des, test_keyp, test_y, cluster_model, self.num_clust, VISUALIZE=False)
            
            X_train.append(X_tr)
            y_train.append(y_tr)
            X_test.append(X_te)
            y_test.append(y_te)

        return X_train, y_train, X_test, y_test

    def get_svm(self):
        # svm_params = {}
        # svm_params['C'] = np.logspace(-2, 3, 6)
        # svm_params['gamma'] = ['scale', 'auto']
        # svm_params['kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']
        svm_params = {}
        svm_params['C'] = [100]
        svm_params['gamma'] = ['scale']
        svm_params['kernel'] = ['rbf']

        mean_accuracies = []
        models = []
        for i in range(len(svm_params['C'])):
            for j in range(len(svm_params['gamma'])):
                for k in range(len(svm_params['kernel'])):
                    # Create a pipeline
                    svc = SVC(C=svm_params['C'][i],
                                                 gamma=svm_params['gamma'][j],
                                                 kernel=svm_params['kernel'][k])
                    models.append(svc)

                    # TODO use something else than accuracy
                    # Get the cross validated accuracy
                    mean_acc = self.get_accuracy_cross_validation(svc, 'SVM')
                    mean_accuracies.append(mean_acc)

        # Store the best model based on the accuracy
        best_model = models[np.argmax(mean_accuracies)]
        print("Best model: ", best_model,
              " with accuracy: ", np.max(mean_accuracies))
        return best_model

    def get_logreg(self):
        # logr_params = {}
        # logr_params['penalty'] = ['l1', 'l2']
        # logr_params['tol'] = np.logspace(-4, -1, 4)
        # logr_params['C'] = np.logspace(-2, 3, 6)
        logr_params = {}
        logr_params['penalty'] = ['l2']
        logr_params['tol'] = [0.0001]
        logr_params['C'] = [1000]

        mean_accuracies = []
        models = []
        for i in range(len(logr_params['penalty'])):
            for j in range(len(logr_params['tol'])):
                for k in range(len(logr_params['C'])):
                    # Create a pipeline
                    logreg = LogisticRegression(
                        penalty=logr_params['penalty'][i], tol=logr_params['tol'][j], C=logr_params['C'][k])
                    models.append(logreg)

                    # TODO use something else than accuracy
                    # Get the cross validated accuracy
                    mean_acc = self.get_accuracy_cross_validation(logreg, "LogReg")
                    mean_accuracies.append(mean_acc)

        # Store the best model based on the accuracy
        best_model = models[np.argmax(mean_accuracies)]
        print("Best model: ", best_model,
              " with accuracy: ", np.max(mean_accuracies))
        return best_model

    def get_rf(self):
        # rf_params = {}
        # rf_params['n_estimators'] = [10, 50, 100]
        # rf_params['max_depth'] = [3, 4, 5, 6, 7]

        rf_params = {}
        rf_params['n_estimators'] = [50]
        rf_params['max_depth'] = [7]

        mean_accuracies = []
        models = []
        for i in range(len(rf_params['n_estimators'])):
            for j in range(len(rf_params['max_depth'])):
                # Create a pipeline
                rf = RandomForestClassifier(
                    n_estimators=rf_params['n_estimators'][i], max_depth=rf_params['max_depth'][j])
                models.append(rf)

                # TODO use something else than accuracy
                # Get the cross validated accuracy
                mean_acc = self.get_accuracy_cross_validation(rf, "RF")
                mean_accuracies.append(mean_acc)
            
        # Store the best model based on the accuracy
        best_model = models[np.argmax(mean_accuracies)]
        print("Best model: ", best_model, " with accuracy: ", np.max(mean_accuracies))
        return best_model

    def get_knn(self):
        # knn_params = {}
        # knn_params['n_neighbors'] = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
        # knn_params['weights'] = ['uniform', 'distance']
        # knn_params['algorithm'] = ['ball_tree', 'kd_tree', 'brute']
        knn_params = {}
        knn_params['n_neighbors'] = [13,]
        knn_params['weights'] = ['distance']
        knn_params['algorithm'] = ['brute']
        
        
        mean_accuracies = []
        models = []
        for i in range(len(knn_params['n_neighbors'])):
            for j in range(len(knn_params['weights'])):
                for k in range(len(knn_params['algorithm'])):
                    # Create a pipeline
                    knn = KNeighborsClassifier(
                        n_neighbors=knn_params['n_neighbors'][i],
                        weights=knn_params['weights'][j],
                        algorithm=knn_params['algorithm'][k])
                    models.append(knn)

                    # TODO use something else than accuracy
                    # Get the cross validated accuracy
                    mean_acc = self.get_accuracy_cross_validation(knn, "KNN")
                    mean_accuracies.append(mean_acc)

        # Store the best model based on the accuracy
        best_model = models[np.argmax(mean_accuracies)]
        print("Best model: ", best_model, " with accuracy: ", np.max(mean_accuracies))
        return best_model

    def train_ensemble(self, models, voting_method='hard'):
        """Train an ensemble of classifiers

        Args:
            models (List(Tuples)): List of tupels (classifiers, name)

        Returns:
            model: Trained ensemble model
        """

        
        # Create an ensemble
        ensemble = VotingClassifier(
            estimators=models,
            voting=voting_method,
        )

        # Cross validate the ensemble
        mean_acc = self.get_accuracy_cross_validation(ensemble, ("Ensemble" + voting_method))
        print('Ensemble scores (', voting_method, '): ', mean_acc)

        return mean_acc, ensemble
