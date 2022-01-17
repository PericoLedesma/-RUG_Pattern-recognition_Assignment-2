import numpy as np
from read_data import *
from data_analysis import *
from feature_extraction import apply_sift
import os.path
from feature_extraction import cluster_sift_descriptions
from feature_extraction import calculate_histogram
from classification import Classifier
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

# Look at the type of errors
# Filter the images (there are many duplicates!)
# Plot of data distribution

# After bug fix (clustering within the cross validation) (No Aug)
# All acc[0.5, 0.4, 0.6, 0.7, 0.4] .    Mean accuracy:  0.52 for SVM
# Best model:  Pipeline(steps=[('svm', SVC(C=100))]) with accuracy:  0.52
# All acc[0.6, 0.6, 0.5, 0.5, 0.5] .    Mean accuracy:  0.54 for LogReg
# Best model:  Pipeline(steps=[('logreg', LogisticRegression(C=1000))]) with accuracy:  0.54
# All acc[0.6, 0.6, 0.4, 0.6, 0.4] .    Mean accuracy:  0.52 for RF
# Best model:  Pipeline(steps=[('rf', RandomForestClassifier(max_depth=7, n_estimators=50))]) with accuracy:  0.52
# All acc[0.6, 0.6, 0.5, 0.6, 0.5] .    Mean accuracy:  0.5599999999999999 for Ensemble
# Ensemble scores:  0.5599999999999999

# After bug fix (With Aug)
# All acc[0.5, 0.45, 0.4, 0.7, 0.6] .   Mean accuracy:  0.53 for SVM
# Best model:  Pipeline(steps=[('svm', SVC(C=100))]) with accuracy:  0.53
# All acc[0.55, 0.65, 0.5, 0.65, 0.55] .        Mean accuracy:  0.5800000000000001 for LogReg
# Best model:  Pipeline(steps=[('logreg', LogisticRegression(C=1000))]) with accuracy:  0.5800000000000001
# All acc[0.55, 0.45, 0.4, 0.6, 0.5] .  Mean accuracy:  0.5 for RF
# Best model:  Pipeline(steps=[('rf', RandomForestClassifier(max_depth=7, n_estimators=50))]) with accuracy:  0.5
# All acc[0.5, 0.55, 0.4, 0.7, 0.65] .  Mean accuracy:  0.56 for Ensemble
# Ensemble scores:  0.56



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
    MIFE = False

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

    # cluster_model = cluster_sift_descriptions(
    #     data['sift_description'], NUM_CLUSTERS=n_clusters)

    # X, y = calculate_histogram(
    #     data['sift_description'], data['sift_keypoints'], data['label'], cluster_model, n_clusters, VISUALIZE=VISUALIZE)

    # #----------Apply PCA----------#
    #plot_pca_components_variance(X)
    # X = apply_pca(X)

    classifier = Classifier(data, num_clust=n_clusters, augment=AUGMENT, debug=DEBUG)
    print("Number of clusters: ", n_clusters)

    # The code below is also executed in the train_ensemble
    svm_model = classifier.get_svm()
    rf_model = classifier.get_rf()
    logreg_model = classifier.get_logreg()
    knn_model = classifier.get_knn()

    accuracy, model = classifier.train_ensemble([(svm_model, 'SVM'), (rf_model, 'RF'), (logreg_model, 'LogReg')],
     voting_method='hard')
    accuracy, model = classifier.train_ensemble([(svm_model, 'SVM'), (rf_model, 'RF'), (logreg_model, 'LogReg')],
     voting_method='soft')

if __name__ == "__main__":
    main()
