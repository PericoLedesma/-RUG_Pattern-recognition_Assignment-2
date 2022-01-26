import numpy as np
from read_data import *
from data_analysis import *
from feature_extraction import apply_sift
import os.path
from feature_extraction import cluster_sift_descriptions
from feature_extraction import calculate_histogram, apply_umap
from classification import Classifier
from feature_extraction import apply_pca
from data_analysis import plot_pca_components_variance
from visualize import sift_bar_plot, plot_umap


def main(n_clusters):
    # Debug mode
    DEBUG  = False
    MAX_DATA = (10 if DEBUG else None)
    # Use data augmentation (not useful when using MIFE)
    AUGMENT = False
    # Create the barplot (and stop the code afterwords)
    BARPLOT = False
    # Create the UMAP plot (and stop the code afterwords)
    UMAP = False
    # Use mirror invariant feature extraction (MIFE)
    # FIXME currently not working (depricated/not used)
    MIFE = False


    pklname_images = f"big_cats{'_augment' if AUGMENT else ''}{'_debug' if DEBUG else ''}.pkl"

    # Load the images
    data = load_data(pklname_images)

    if data is None:
        # MAX_DATA indicates the number of images to be read from each class
        # Since this is used for testing, the pickle file is created with the addition '_debug'
        data = main_read_data(pklname_images, max_data=MAX_DATA)

    # Sift feature extraction
    data = apply_sift(data, mife=MIFE)

    if BARPLOT:
        feature_vector, label_vector = sift_bar_plot(data, n_clusters)
        exit()

    if UMAP:
        feature_vector, label_vector = sift_bar_plot(data, n_clusters)
        umap_data = apply_umap(feature_vector)
        plot_umap(umap_data, label_vector, n_clusters)
        exit()

    classifier = Classifier(data, num_clust=n_clusters, augment=AUGMENT, debug=DEBUG)
    print("Number of clusters: ", n_clusters)

    # Create and train the individual classifiers
    svm_model, svm_acc, svm_params = classifier.get_svm()
    rf_model, rf_acc, rf_params = classifier.get_rf()
    logreg_model, logreg_acc, logreg_params = classifier.get_logreg()
    knn_model, knn_acc, knn_params = classifier.get_knn()
    
    # Creating the ensemble classifier
    ensemble, acc_ensemble = classifier.train_ensemble([('SVM', svm_model), ('RF', rf_model), ('LogReg', logreg_model)],
     voting_method='hard')

    print ("SVM accuracy: ", svm_acc, " with params: ", svm_params)
    print ("RF accuracy: ", rf_acc, " with params: ", rf_params)
    print ("LogReg accuracy: ", logreg_acc, " with params: ", logreg_params)
    print ("KNN accuracy: ", knn_acc, " with params: ", knn_params)
    print ("Ensemble scores: ", acc_ensemble)

if __name__ == "__main__":
    # Loop for gird search clusters
    # for i in range(10, 60, 10):
    main(n_clusters=30)
