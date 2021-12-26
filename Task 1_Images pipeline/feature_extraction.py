import cv2
import numpy as np
import joblib

# ---- Feature extraction  ----

# -- SIFT --
# https://www.analyticsvidhya.com/blog/2019/10/detailed-guide-powerful-sift-technique-image-matching-python/
def apply_sift(data):
    for img in data['data']:
        img = np.float32(img)
        # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Binarization
        img = cv2.normalize(img, None, 0, 255,
                            cv2.NORM_MINMAX).astype('uint8')

        # Get the sift keypoints and the corresponding descriptors
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(img, None)
        if 'sift_keypoints' not in data.keys():
            print('created list')
            data['sift_keypoints'] = []
        data['sift_keypoints'].append(kp)
    return data

if __name__ == '__main__':
    base_name = 'big_cats'
    width = 80
    data = joblib.load(f'{base_name}_{width}x{width}px.pkl')
    data = apply_sift(data)
    # Temporary print loop (to analyze the data)
    # for keypoints in data['sift_keypoints'][:5]:
    #     print('\n')
    #     for d in keypoints:
    #         print(str(d.angle) + ' ' + str(d.class_id) + ' ' + str(d.octave) + ' ' + str(d.pt)  + ' ' +
    #         str(d.response) + ' ' + str(d.size))
