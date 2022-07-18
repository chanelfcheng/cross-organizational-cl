import os
import glob
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from scipy.stats.mstats import winsorize
from torch.utils.data import TensorDataset
from utils.data_preprocessing import process_features, replace_invalid, resample_data, CIC_2018, USB_2021

def load_datasets(dset, data_path, pkl_path, include_categorical):
    all_features = None
    all_labels = []
    all_invalid = 0

    if os.path.exists(pkl_path): 
        with open(pkl_path, 'rb') as file:
            features_train, features_test, labels_train, labels_test = pickle.load(file)  # Load data from pickle file
    else:
        for file in list(glob.glob(f'{data_path}/*.csv')):
            print('Loading ', file, '...')
            reader = pd.read_csv(file, dtype=str, chunksize=10**6, skipinitialspace=True)  # Read in data from csv file

            for df in reader:
                features, labels = process_features(dset, df, include_categorical)

                # Convert dataframe to numpy array for processing
                data_np = np.array(features.to_numpy(), dtype=float)
                labels_lst = labels.tolist()

                data_np, labels_lst, num_invalid = replace_invalid(data_np, labels_lst)  # Clean data of invalid values

                # Combine all data, labels, and number of invalid values
                if all_features is None:
                    all_features = data_np  # If no data yet, set all data to current data
                else:
                    all_features = np.concatenate((all_features, data_np))  # Else, concatenate data
                all_labels += labels_lst
                all_invalid += num_invalid

        # Print total number of invalid values dropped, total number of data
        # values, and total percentage of invalid data
        print('Total Number of invalid values: %d' % all_invalid)
        print('Total Data values: %d' % len(all_labels))
        print('Invalid data: %.2f%%' % (all_invalid / float(all_features.size) * 100))

        # Save histogram of cleaned data
        axs = pd.DataFrame(all_features, columns=features.columns.values.tolist()).hist(figsize=(30,30))
        plt.tight_layout()
        plt.savefig(dset + '-hist.png')

        # Perform train/test split of 80-20
        features_train, features_test, labels_train, labels_test = train_test_split(all_features, all_labels, test_size=0.2)

        # Resample training data to reduce class imbalance
        features_train, labels_train = resample_data(dset, features_train, labels_train)
        
        # Save to pickle file
        with open(pkl_path, 'wb') as file:
            pickle.dump((features_train, features_test, labels_train, labels_test), file)
        
    return features_train, features_test, labels_train, labels_test

def load_pytorch_datasets(dset, data_path, pkl_path, include_categorical, model='mlp'):
    # Load in datasets from csv files
    features_train, features_test, labels_train, labels_test = load_datasets(dset, data_path, pkl_path, include_categorical)

    # Normalize train and test data
    scale = RobustScaler(quantile_range=(5,95)).fit(features_train)
    features_train = scale.transform(features_train)
    features_test = scale.transform(features_test)

    # Create pytorch datasets for data only
    features_train = torch.tensor(features_train)
    features_test = torch.tensor(features_test)

    # Reshape input features for CNN
    if model == 'cnn':
        features_train = features_train.reshape(len(features_train), features_train.shape[1], 1)
        features_test = features_test.reshape(len(features_test), features_test.shape[1], 1)
        features_train.shape, features_test.shape

    # Label encoding
    label_encoding = {}
    value = 0
    for label in labels_test:
        if label not in label_encoding:
            label_encoding[label] = value
            value += 1
    
    labels_idfeatures_train = []
    for i in range(len(labels_train)):
        label = labels_train[i]
        value = label_encoding[label]
        labels_idfeatures_train.append(value)

    labels_idfeatures_test = []
    for i in range(len(labels_test)):
        label = labels_test[i]
        value = label_encoding[label]
        labels_idfeatures_test.append(value)

    labels_train = torch.tensor(labels_idfeatures_train)
    labels_test = torch.tensor(labels_idfeatures_test)
    classes = list(label_encoding.keys())

    # Create pytorch datasets with labels
    dataset_train = TensorDataset(features_train, labels_train)
    dataset_test = TensorDataset(features_test, labels_test)

    # Define attack classes
    dataset_train.classes = classes
    dataset_test.classes = classes

    return dataset_train, dataset_test

def main():
    # features_train, features_test, labels_train, labels_test = load_datasets(
    #     dset=CIC_2018, 
    #     data_path='/home/chanel/Cyber/yang-summer-2022/data/CIC-IDS2018/Hulk-Slowloris-Slowhttptest', 
    #     pkl_path='/home/chanel/Cyber/yang-summer-2022/cross-organizational-cl/pickle/cic-2018.pkl',
    #     include_categorical=True
    # )
    # features_train, features_test, labels_train, labels_test = load_datasets(
    #     dset=USB_2021, 
    #     data_path='/home/chanel/Cyber/yang-summer-2022/data/USB-IDS2021', 
    #     pkl_path='/home/chanel/Cyber/yang-summer-2022/cross-organizational-cl/pickle/usb-2021.pkl',
    #     include_categorical=True
    # )
    # features_train, features_test, labels_train, labels_test = load_datasets(
    #     dset=CIC_2018, 
    #     data_path='/home/chanel/Cyber/yang-summer-2022/data/CIC-IDS2018/Hulk-Slowloris-Slowhttptest', 
    #     pkl_path='/home/chanel/Cyber/yang-summer-2022/cross-organizational-cl/pickle/cic-2018-no-categorical.pkl',
    #     include_categorical=False
    # )
    features_train, features_test, labels_train, labels_test = load_datasets(
        dset=USB_2021, 
        data_path='/home/chanel/Cyber/yang-summer-2022/data/USB-IDS2021', 
        pkl_path='/home/chanel/Cyber/yang-summer-2022/cross-organizational-cl/pickle/usb-2021-no-categorical.pkl',
        include_categorical=False
    )
    print(features_train.shape)

if __name__ == '__main__':
    main()