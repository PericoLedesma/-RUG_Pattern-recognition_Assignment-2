from feature_extraction import cluster_sift_descriptions
from feature_extraction import calculate_histogram

def sift_bar_plot(data, n_clusters):

    model = cluster_sift_descriptions(data['sift_description'], n_clusters)
    calculate_histogram(data['sift_description'],
                        data['sift_keypoints'], data['label'], model, n_clusters, VISUALIZE=True)

    exit()
