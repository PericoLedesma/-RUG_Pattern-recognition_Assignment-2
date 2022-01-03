#This implementation is heavily based on
#https://kapernikov.com/tutorial-image-classification-with-scikit-learn/
import matplotlib.pyplot as plt
import numpy as np
import os
import pprint
from collections import Counter
pp = pprint.PrettyPrinter(indent = 4)
import joblib
from data_analysis import analyze_data
from skimage.io import imread
from skimage.transform import resize
import os

def read_data(src, pklname, include, width = 150, height = None):

    height = height if height is not None else width

    data = dict()
    data['description'] = 'resized ({0}x{1}) big cat images in rgb'.format(int(width), int(height))
    data['label'] = []
    data['filename'] = []
    data['data'] = []

    pklname = f"{pklname}_{width}x{height}px.pkl"

    #Read all images in PATH, resize and write to DESTIONATION_PATH
    for subdir in os.listdir(src):
        if subdir in include:
            current_path = os.path.join(src, subdir)

            for file in os.listdir(current_path):
                if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg'):
                    im = imread(os.path.join(current_path, file))
                    # im = resize(im, (width, height))
                    data['label'].append(subdir)
                    data['filename'].append(file)
                    data['data'].append(im)
        joblib.dump(data, pklname)

def print_summary(data):
    print('number of samples: ', len(data['data']))
    print('labels', np.unique(data['label']))
    print('description: ', data['description'])

def main_read_data():
    data_path = 'data/BigCats/' #Specify data path
    classes = os.listdir(data_path) #Define classes
    if '.DS_Store' in classes: classes.remove('.DS_Store') #For Mac Users, unwanted folder

    base_name = 'big_cats'
    width = 0
    include = {'Leopard', 'Tiger', 'Cheetah', 'Jaguar', 'Lion'} #Include the wanted classes

    # Read the data
    read_data(src = data_path, pklname = base_name, width = width, include = include)
    data = joblib.load(f'{base_name}_{width}x{width}px.pkl')
    return data
