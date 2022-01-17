import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def apply_pca(train_featvec):
    """This function applies PCA on the features
    Args:
        train_featvec (List): List of the feature vectors
    Returns:
        pca_data: The data transformed in the dimensions with the highest explained variance
    """
    pca = PCA(n_components = 80)
    pca.fit(train_featvec)
    pca_data = pca.transform(train_featvec)

    return pca_data


def cluster_sift_descriptions(sift_des, NUM_CLUSTERS):
    # Prepare data
    print("Training kmeans")
    # Create model
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=0)
    # train model
    sift_descriptors = np.concatenate(sift_des, axis=0)
    kmeans.fit(sift_descriptors)
    return kmeans


def get_class(animal):
    class_as_number = 0
    if animal == 'Cheetah':
        class_as_number = 1
    if animal == 'Jaguar':
        class_as_number = 2
    if animal == 'Leopard':
        class_as_number = 3
    if animal == 'Lion':
        class_as_number = 4
    if animal == 'Tiger':
        class_as_number = 5
    if class_as_number == 0:
        print('ANIMAL NAME NOT DEFINED!\n')
        print(animal)
    return class_as_number

def get_name_by_number(class_number):
    if class_number == 1:
        return 'Cheetah'
    if class_number == 2:
        return 'Jaguar'
    if class_number == 3:
        return 'Leopard'
    if class_number == 4:
        return 'Lion'
    if class_number == 5:
        return 'Tiger'
    return 'UNKNOWN'


def calculate_histogram(sift_des, sift_keyp, labels, model, n_clusters, VISUALIZE=False):
    feature_vector = []
    label_vector = []

    # Make prediction to what class the keypoint belong and make a histogram of that
    idx = 0
    for img_des in sift_des:
        # Predict and create histogram
        predict_kmeans = model.predict(img_des)
        hist, bin_edges = np.histogram(predict_kmeans, bins=n_clusters)
        # Normalize the histogram
        hist = hist / len(sift_keyp[idx])
        feature_vector.append(hist)

        # Create target vector for classification
        label = get_class(labels[idx])
        label_vector.append(label)
        idx = idx + 1

    if VISUALIZE:
        all_histograms = []
        cur_label = label_vector[0]
        hist_labels = [cur_label]
        avg_hist = np.zeros(len(feature_vector[0]))
        # Loop through the label_vector
        for i in range(len(label_vector)):
            if label_vector[i] == cur_label:
                avg_hist = avg_hist + feature_vector[i]
            else:
                # Count the appearance of cur_label in the label_vector
                all_histograms.append(avg_hist/label_vector.count(cur_label))
                cur_label = label_vector[i]
                hist_labels.append(cur_label)
                avg_hist = feature_vector[i]
        all_histograms.append(avg_hist/label_vector.count(cur_label))

        # Plot the histograms in a single barplot next to each other
        ind = np.arange(len(all_histograms[0]))
        width = 0.15
        fig, ax = plt.subplots()
        for i in range(len(all_histograms)):
            ax.bar(ind + (i * width), all_histograms[i], width, label=get_name_by_number(hist_labels[i]))
        ax.set_xticks(ind + width*0.5)
        ax.set_ylabel('Frequency')
        ax.set_xlabel('Cluster')
        ax.set_title('Histogram of SIFT keypoints')
        ax.set_xticks(ind)
        ax.legend()
        plt.show()

    return feature_vector, label_vector


# -- SIFT --
# https://www.analyticsvidhya.com/blog/2019/10/detailed-guide-powerful-sift-technique-image-matching-python/
def apply_sift(data, mife=False):
    for img in data['image']:
        img = np.float32(img)
        # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Binarization
        img = cv2.normalize(img, None, 0, 255,
                            cv2.NORM_MINMAX).astype('uint8')

        # Get the sift keypoints and the corresponding descriptors
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(img, None)
        if 'sift_description' not in data.keys():
            data['sift_description'] = []
        if 'sift_keypoints' not in data.keys():
            data['sift_keypoints'] = []

        # Extract features from the mirrored image
        if mife:
            mirrored_image = np.fliplr(img)
            sift = cv2.SIFT_create()
            kp_mife, des_mife = sift.detectAndCompute(mirrored_image, None)
            kp = np.concatenate((kp, kp_mife), axis=0)
            des = np.concatenate((des, des_mife), axis=0)
        data['sift_description'].append(des)
        data['sift_keypoints'].append(kp)

    return data


if __name__ == '__main__':
    base_name = 'big_cats'
    width = 0
    data = joblib.load(f'{base_name}_{width}x{width}px.pkl')
    data = apply_sift(data)
    # Temporary print loop (to analyze the data)
    # for keypoints in data['sift_keypoints'][:5]:
    #     print('\n')
    #     for d in keypoints:
    #         print(d)
    #         # print(str(d.angle) + ' ' + str(d.class_id) + ' ' + str(d.octave) + ' ' + str(d.pt)  + ' ' +
    #         # str(d.response) + ' ' + str(d.size))
