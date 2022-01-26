from feature_extraction import cluster_sift_descriptions
from feature_extraction import calculate_histogram, get_name_by_number
import matplotlib.pyplot as plt
import numpy as np

def sift_bar_plot(data, n_clusters):

    model = cluster_sift_descriptions(data['sift_description'], n_clusters)
    feature_vector, label_vector = calculate_histogram(data['sift_description'],
                        data['sift_keypoints'], data['label'], model, n_clusters, VISUALIZE=True)
    return feature_vector, label_vector

def plot_umap(umap_data, labels, n_clusters):
    # Plot UMAP
    plt.figure(figsize=(10, 10))
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    cats = ['Cheetah', 'Jaguar', 'Leopard', 'Lion', 'Tiger']

    # Reorder botht he umap_data and labels in the same order
    umap_data = [umap_data[i] for i in np.argsort(labels)]
    labels = [labels[i] for i in np.argsort(labels)]

    fig, ax = plt.subplots()
    for idx, lab in enumerate(labels):
        ax.scatter(umap_data[idx][0], umap_data[idx][1],
                   c=colors[lab-1], label=cats[lab-1])
    # ax.legend()

    # https://stackoverflow.com/questions/19385639/duplicate-items-in-legend-in-matplotlib
    legend_without_duplicate_labels(ax)
    # Title
    plt.title('UMAP projection of clustered SIFT keypoints (n_clusters={})'.format(n_clusters))
    # X-axis label
    plt.xlabel('First UMAP dimension')
    # Y-axis label
    plt.ylabel('Second UMAP dimension')
    plt.show()


    # label_colors = [colors[i-1] for i in labels]
    # plt.scatter([x for x, y in umap_data], [y for x, y in umap_data], c=label_colors, cmap='tab10')
    # # add a legend
    
    import pdb; pdb.set_trace()


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(
        zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

