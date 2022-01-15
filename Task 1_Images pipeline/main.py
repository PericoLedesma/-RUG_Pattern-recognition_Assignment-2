import numpy as np
from read_data import *
from data_analysis import *
from feature_extraction import apply_sift
import os.path
from feature_extraction import cluster_sift_descriptions
from classification import train_model
from feature_extraction import calculate_histogram
from classification import train_SVM_model, train_KNN_model, train_RF_model, get_accuracy_cross_validation
from feature_extraction import apply_pca
from data_analysis import plot_pca_components_variance
from data_augmentation import augment_data

width = 80
base_name = 'big_cats'

n_clusters = 20



# WITH MIFE
# The best parameters for SVM are {'C': 100.0, 'gamma': 'scale', 'kernel': 'rbf'} with a score of 0.60
# The best parameters for Logistic Regression are {'C': 1000.0, 'penalty': 'l2', 'tol': 0.0001} with a score of 0.54
# The best parameters for Random Forest are {'max_depth': 7, 'n_estimators': 50} with a score of 0.55
# Ensemble scores:  [0.61764706 0.64705882 0.61764706 0.64705882 0.52941176]
# Ensemble mean score:  0.6117647058823529
# Number of clusters:  20
# accuracy rf = 0.5
# accuracy svm = 0.5235294117647059
# accuracy knn = 0.5294117647058824

# TODO PCA for Kmeans
# TODO Other classifiers/ ensemble methods
# TODO Visualize histogram/cluters
# TODO SetScore


# KNN
# MIRROR in feature extraction
# Ensemble met random forest


def main():
    # Debug mode
    DEBUG  = True
    MAX_DATA = (10 if DEBUG else None)
    # Visualize the data in plots
    VISUALIZE = True
    # Use data augmentation (not useful when using MIFE)
    # NOTE This should not be used. Augmentation should not be applied to test data
    AUGMENT = True
    # Use mirror invariant feature extraction (MIFE)
    MIFE = True

    pklname_images = f"big_cats{'_augment' if AUGMENT else ''}{'_debug' if DEBUG else ''}.pkl"

    # Load the images
    data = load_data(pklname_images)

    if data is None:
        # NOTE MAX_DATA indicates the number of images to be read from each class
        # Since this is used for testing, the pickle file is created with the addition '_test'
        data = main_read_data(pklname_images, max_data=MAX_DATA)

        # Print a summary of the data based on user input
        #print_summary = input("Do you want to print a summary of the data and see the images? Yes: 1, No: 0\n")
        #if print_summary == "1": analyze_data(data)

    # Start feature extraction
    #--- SIFT --- #
    data = apply_sift(data, mife=MIFE)

    cluster_model = cluster_sift_descriptions(data, NUM_CLUSTERS=n_clusters)

    X, y = calculate_histogram(
        data, cluster_model, n_clusters, VISUALIZE=VISUALIZE)

    #----------Apply PCA----------#
    #plot_pca_components_variance(X)
    # X = apply_pca(X)

    # Train an ensemble of classifiers
    accuracy, model = train_model(
        X, y, cluster_model=cluster_model, data=data, n_models=10, DEBUG=DEBUG, augment=AUGMENT)
    svm_model = train_SVM_model(
        X, y, cluster_model=cluster_model, data=data, augment=AUGMENT)
    rf_model = train_RF_model(
        X, y, cluster_model=cluster_model, data=data, augment=AUGMENT)
    knn_model = train_KNN_model(
        X, y, cluster_model=cluster_model, data=data, augment=AUGMENT)

    accuracy_rf = get_accuracy_cross_validation(rf_model, X, y)
    accuracy_svm = get_accuracy_cross_validation(svm_model, X, y)
    accuracy_knn = get_accuracy_cross_validation(knn_model, X, y)
    print("Number of clusters: ", n_clusters)
    print("accuracy rf = ",  accuracy_rf)
    print("accuracy svm = ",  accuracy_svm)
    print("accuracy knn = ",  accuracy_knn)


if __name__ == "__main__":
    main()
