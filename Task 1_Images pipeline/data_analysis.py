from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def plot_pca_components_variance(train_featvec):
    pca = PCA().fit(train_featvec)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    #plt.show()
    plt.savefig('figures/100_clusters_explained_variance.png')

def analyze_data(data):
    """Analyze the data

    Args:
        data (dict): Dictionary created by read_data.py
    """

    # Images per class
    c = Counter(data['label'])
    print("Images per class: ", c)
    print(np.unique(data['label']))

    # Loop through all the different classes
    for img_class in np.unique(data['label']):
        print('images of class: ', img_class)
        # Loop through all the images of that class
        counter = 0
        # Create a 5x5 grid of images
        fig, ax = plt.subplots(5, 5, figsize=(10, 10))
        for idx, img in enumerate(data['sift_description']):
            # Show the images of that class
            if data["label"][idx] == img_class:
                i = int(counter / 5)
                j = counter % 5
                ax[i, j].imshow(img)
                ax[i, j].axis('off')
                counter += 1
                if counter >= 25:
                    break
        plt.show()
