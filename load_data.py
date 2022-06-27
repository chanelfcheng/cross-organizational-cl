import os
import glob
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from utils.data_preprocessing import resample_data, process_features, replace_invalid, CIC_2018, USB_2021

def load_datasets(name, data_path, pkl_path):
    all_data = None
    all_labels = []
    all_invalid = 0

    if os.path.exists(pkl_path): 
        with open(pkl_path, 'rb') as file:
            data_train, data_test, labels_train, labels_test = pickle.load(file)  # Load data from pickle file
    else:
        for file in list(glob.glob(f'{data_path}/*.csv')):
            print('Loading ', file, '...')
            reader = pd.read_csv(file, dtype=str, chunksize=10**6, skipinitialspace=True)  # Read in data from csv file

            for df in reader:
                data, labels = process_features(name, df)

                # Convert dataframe to numpy array for processing
                data_np = np.array(data.to_numpy(), dtype=float)
                labels_lst = labels.tolist()

                data_np, labels_lst, num_invalid = replace_invalid(data_np, labels_lst)  # Clean the data

                # Combine all data, labels, and number of invalid values
                if all_data is None:
                    all_data = data_np  # If no data yet, set all data to current data
                else:
                    all_data = np.concatenate((all_data, data_np))  # Else, concatenate data
                all_labels += labels_lst
                all_invalid += num_invalid

        # Print total number of invalid values dropped, total number of data
        # values, and total percentage of invalid data
        print('Total Number of invalid values: %d' % all_invalid)
        print('Total Data values: %d' % len(all_labels))
        print('Invalid data: %.2f%%' % (all_invalid / float(all_data.size) * 100))

        # Save histogram of cleaned data
        axs = pd.DataFrame(all_data, columns=data.columns.values.tolist()).hist(figsize=(30,30))
        plt.tight_layout()
        plt.savefig(name + '-hist.png')

        # Perform train/test split of 80-20
        data_train, data_test, labels_train, labels_test = train_test_split(all_data, all_labels, test_size=0.2)

        # Resample data to reduce class imbalance
        data_train, labels_train = resample_data(name, data_train, labels_train)

        # Save to pickle file
        with open(pkl_path, 'wb') as file:
            pickle.dump((data_train, data_test, labels_train, labels_test), file)
        
    return data_train, data_test, labels_train, labels_test

def load_pytorch_datasets(name, data_path, pkl_path=None):
    data_train, data_test, labels_train, labels_test = load_datasets(name, data_path, pkl_path)
    data_train = torch.tensor(data_train)
    data_test = torch.tensor(data_test)

    label_encoding = {}
    value = 0
    for label in labels_test:
        if label not in label_encoding:
            label_encoding[label] = value
            value += 1
    
    labels_idx_train = []
    for i in range(len(labels_train)):
        label = labels_train[i]
        value = label_encoding[label]
        labels_idx_train.append(value)

    labels_idx_test = []
    for i in range(len(labels_test)):
        label = labels_test[i]
        value = label_encoding[label]
        labels_idx_test.append(value)

    labels_train = torch.tensor(labels_idx_train)
    labels_test = torch.tensor(labels_idx_test)
    classes = list(label_encoding.keys())

    dataset_train = TensorDataset(data_train, labels_train)
    dataset_test = TensorDataset(data_test, labels_test)

    dataset_train.classes = classes
    dataset_test.classes = classes

    return dataset_train, dataset_test

def main():
    # load_data(
    #     name=CIC_2018, 
    #     data_path='/home/chanel/Cyber/yang-summer-2022/data/CIC-IDS2018/Hulk-Slowloris', 
    #     pkl_path='/home/chanel/Cyber/yang-summer-2022/cross-organizational-cl/pickle/cic-2018.pkl'
    # )
    load_datasets(
        name=USB_2021, 
        data_path='/home/chanel/Cyber/yang-summer-2022/data/USB-IDS2021/Hulk-Slowloris', 
        pkl_path='/home/chanel/Cyber/yang-summer-2022/cross-organizational-cl/pickle/usb-2018.pkl'
    )

if __name__ == '__main__':
    main()