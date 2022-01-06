import numpy as np
from read_data import *
from data_analysis import *
from feature_extraction import apply_sift
import os.path
from feature_extraction import cluster_sift_descriptions
from classification import train_model
from feature_extraction import calculate_histogram

width = 80
base_name = 'big_cats'

n_clusters = 20

# TODO Normalize the histogram (divide by the number of keypoints)
# TODO Add KP to data object
# TODO PCA for Kmeans
# TODO Other classifiers/ ensemble methods
# TODO Visualize histogram/cluters
# TODO SetScore


def main():
    # Debug mode
    DEBUG  = False
    # Visualize the data in plots
    VISUALIZE = False

    # NOTE MAX_DATA indicates the number of images to be read from each class
    # Since this is used for testing, the pickle file is created with the addition '_test'
    data = main_read_data(MAX_DATA=(10 if DEBUG else None))

    # Print a summary of the data based on user input
    #print_summary = input("Do you want to print a summary of the data and see the images? Yes: 1, No: 0\n")
    #if print_summary == "1": analyze_data(data)

    # Start feature extraction

    #--- SIFT --- #
    data = apply_sift(data)

    model = cluster_sift_descriptions(data, NUM_CLUSTERS=n_clusters)

    X, y = calculate_histogram(data, model, VISUALIZE=VISUALIZE)

    # Train an ensemble of classifiers
    accuracy, model = train_model(X, y, n_models=10, DEBUG=DEBUG)
    print(accuracy)


if __name__ == "__main__":
    main()
