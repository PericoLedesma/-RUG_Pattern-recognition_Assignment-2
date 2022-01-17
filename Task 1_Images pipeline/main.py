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

# TODO PCA for Kmeans
# TODO Visualize histogram/clusters
# TODO SetScore
# TODO Look at the type of errors
# TODO Filter the images (there are many duplicates!)
# TODO Plot of data distribution
# TODO Other plots for data visualization/analysis
# TODO Consider other metrics (besides accuracy)

# Using all data, without any augmentation
# SVM accuracy:  0.5823529411764706 with params:  [100.0, 'scale', 'rbf']
# RF accuracy:  0.5294117647058824 with params:  [10, 6]
# LogReg accuracy:  0.5588235294117647 with params:  ['l2', 0.0001, 1000.0]
# KNN accuracy:  0.5470588235294118 with params:  [11, 'distance', 'ball_tree']
# Ensemble scores:  0.6294117647058824

# Using all data, with data augmentation
# SVM accuracy:  0.5882352941176471 with params:  [0.1, 'scale', 'poly']
# RF accuracy:  0.5411764705882354 with params:  [50, 7]
# LogReg accuracy:  0.5558823529411765 with params:  ['l2', 0.0001, 1000.0]
# KNN accuracy:  0.5205882352941176 with params:  [1, 'uniform', 'ball_tree']
# Ensemble scores:  0.5558823529411765

def main():
    # Debug mode
    DEBUG  = True
    MAX_DATA = (10 if DEBUG else None)
    # Use data augmentation (not useful when using MIFE)
    AUGMENT = False
    # Use mirror invariant feature extraction (MIFE)
    # FIXME currently not working
    MIFE = False

    n_clusters = 20
    pklname_images = f"big_cats{'_augment' if AUGMENT else ''}{'_debug' if DEBUG else ''}.pkl"

    # Load the images
    data = load_data(pklname_images)

    if data is None:
        # MAX_DATA indicates the number of images to be read from each class
        # Since this is used for testing, the pickle file is created with the addition '_debug'
        data = main_read_data(pklname_images, max_data=MAX_DATA)

    # Sift feature extraction
    data = apply_sift(data, mife=MIFE)

    classifier = Classifier(data, num_clust=n_clusters, augment=AUGMENT, debug=DEBUG)
    print("Number of clusters: ", n_clusters)

    # The code below is also executed in the train_ensemble
    svm_model, svm_acc, svm_params = classifier.get_svm()
    rf_model, rf_acc, rf_params = classifier.get_rf()
    logreg_model, logreg_acc, logreg_params = classifier.get_logreg()
    knn_model, knn_acc, knn_params = classifier.get_knn()

    ensemble, acc_ensemble = classifier.train_ensemble([('SVM', svm_model), ('RF', rf_model), ('LogReg', logreg_model)],
     voting_method='hard')
    # TODO Look at the possibility to change voting method to soft. This requires probability predictions (other y-values)
    # accuracy_soft, model_soft = classifier.train_ensemble([('SVM', svm_model), ('RF', rf_model), ('LogReg', logreg_model)],
    #  voting_method='soft')

    print ("SVM accuracy: ", svm_acc, " with params: ", svm_params)
    print ("RF accuracy: ", rf_acc, " with params: ", rf_params)
    print ("LogReg accuracy: ", logreg_acc, " with params: ", logreg_params)
    print ("KNN accuracy: ", knn_acc, " with params: ", knn_params)
    print ("Ensemble scores: ", acc_ensemble)

if __name__ == "__main__":
    main()
