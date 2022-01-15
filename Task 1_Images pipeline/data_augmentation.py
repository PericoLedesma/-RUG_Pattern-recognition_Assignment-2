import numpy as np
import matplotlib.pyplot as plt

def augment(data):
    # Augment the data
    for i in range(len(data['image'])):
        # Mirror the image
        mirrored_image = np.fliplr(data['image'][i])
        # Add the mirrored image to the data
        data['label'].append(data['label'][i])
        data['image'].append(mirrored_image)
        data['filename'].append('augmented_' + data['filename'][i])
    return data
