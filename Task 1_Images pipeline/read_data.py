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

def read_data(src, pklname, include, max_data = None):

    data = dict()
    data['description'] = 'big cat images in rgb'
    data['label'] = []
    data['filename'] = []
    data['image'] = []

    # Check if the data is already read
    if os.path.isfile(pklname):
        print(f'File {pklname} already exists. Reading data from pickle file...')
        data = joblib.load(pklname)
        return data

    print("Reading images")
    #Read all images in PATH, resize and write to DESTIONATION_PATH
    for subdir in os.listdir(src):
        if subdir in include:
            current_path = os.path.join(src, subdir)
            counter = 0
            for file in os.listdir(current_path):
                if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg'):
                    counter += 1
                    im = imread(os.path.join(current_path, file))
                    data['label'].append(subdir)
                    data['filename'].append(file)
                    data['image'].append(im)
                    if max_data:
                        if counter >= max_data:
                            break
    print("Done reading images")
    return data


def load_data(pklname):
    data = None
    # Check if the data is already read
    if os.path.isfile(pklname):
        print(f'File {pklname} already exists. Reading data from pickle file...')
        data = joblib.load(pklname)
        return data

def save_data(data, pklname):
    joblib.dump(data, pklname)

def print_summary(data):
    print('number of samples: ', len(data['image']))
    print('labels', np.unique(data['label']))
    print('description: ', data['description'])

def main_read_data(pklname, max_data = None):
    data_path = 'BigCats_better/' #Specify data path
    classes = os.listdir(data_path) #Define classes
    if '.DS_Store' in classes: classes.remove('.DS_Store') #For Mac Users, unwanted folder

    include = {'Leopard', 'Tiger', 'Cheetah', 'Jaguar', 'Lion'} #Include the wanted classes

    # Read the data
    data = read_data(src = data_path, pklname = pklname, include = include, max_data = max_data)
    return data
