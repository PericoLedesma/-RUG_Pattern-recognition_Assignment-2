import numpy as np
from read_data import *
from data_analysis import *
from feature_extraction import apply_sift
import os.path
from feature_extraction import cluster_sift_descriptions
from classification import do_svm
from feature_extraction import calculate_histogram

width = 80
base_name = 'big_cats'

n_clusters = 20

def main():
    #Read the data
    print("Reading data...")

    data = main_read_data()

    #Print a summary of the data based on user input
    #print_summary = input("Do you want to print a summary of the data and see the images? Yes: 1, No: 0\n")
    #if print_summary == "1": analyze_data(data)

    #Start feature extraction

    #--- SIFT --- #
    data = apply_sift(data)

    model = cluster_sift_descriptions(data, NUM_CLUSTERS = n_clusters)

    X, y = calculate_histogram(data, model)

    #Train a simple SVM
    accuracy = do_svm(X, y)
    print(accuracy)





if __name__ == "__main__":
    main()
