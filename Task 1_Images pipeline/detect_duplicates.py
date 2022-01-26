import numpy as np
import os
import collections

def load_data(pklname):
    data = None
    # Check if the data is already read
    if os.path.isfile(pklname):
        print(f'File {pklname} already exists. Reading data from pickle file...')
        data = joblib.load(pklname)
        return data

folders = os.listdir("data/BigCats/")

if '.DS_Store' in folders: folders.remove('.DS_Store') #For Mac Users, unwanted folder

all_files = []
for folder in folders:
    files = os.listdir("data/BigCats/" + folder)
    for file in files:
        split_string = file.split(".", 1)
        file_name = split_string[0]

        all_files.append(file_name)



print([item for item, count in collections.Counter(all_files).items() if count > 1])
