import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from imblearn.over_sampling import RandomOverSampler, SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter
from utils.compare_features import get_attribute_map

CIC_2018 = 'cic-2018'
USB_2021 = 'usb-2021'

def process_features(dset, df):
    print('Processing features...')
    # Rename attribute and benign label names to be consistent with CIC 2018 dataset
    if dset == USB_2021:
        attribute_map = get_attribute_map()
        df = df.rename(columns=attribute_map, errors='raise')
        df['Label'] = df['Label'].replace('BENIGN', 'Benign')

    attack = df.loc[df['Label'].str.contains('hulk|slowloris', case=False)].copy()  # Only get hulk/slowloris attacks
    benign = df.loc[df['Label'].str.contains('benign', case=False)].copy()  # Get all benign traffic

    data = pd.concat([attack, benign]).drop(['Label'], axis=1)  # Concatenate attack/benign and separate label from data
    labels = pd.concat([attack, benign])['Label']  # Save labels by themselves

    # Remove unused columns if present
    if 'Timestamp' in data:
        data = data.drop('Timestamp', axis=1)
    if 'Flow ID' in data:
        data = data.drop('Flow ID', axis=1)
    if 'Src IP' in data:
        data = data.drop('Src IP', axis=1)
    if 'Src Port' in data:
        data = data.drop('Src Port', axis=1)
    if 'Dst IP' in data:
        data = data.drop('Dst IP', axis=1)

    # Protocol one-hot encoding
    print('protocol one-hot encoding...')
    le1 = LabelEncoder()
    le_protocol = le1.fit_transform(data['Protocol'])

    le_protocol = le_protocol.reshape(len(le_protocol), 1)
    ohe1 = OneHotEncoder(sparse=False)
    ohe_protocol = ohe1.fit_transform(le_protocol)
    ohe_protocol = pd.DataFrame(ohe_protocol).rename(columns={0:'Protocol 0', 1:'Protocol 6', 2:'Protocol 17'})
    
    data.reset_index(drop=True, inplace=True)
    labels.reset_index(drop=True, inplace=True)

    data = data.drop('Protocol', axis=1)
    data = data.join(ohe_protocol)

    # Destination port frequency encoding with aggregation
    print('destination port aggregate frequency encoding...')
    le2 = LabelEncoder()
    agg_dport, agg_unique_dport = cumulatively_categorise(data['Dst Port'], threshold=0.90)
    le_agg_dport = le2.fit_transform(agg_dport)
    le_agg_dport = pd.Series(le_agg_dport, name='Dst Port')

    # print('aggregate one-hot encoding')
    # dport_dict = {}
    # for dport in np.unique(le_agg_dport):
    #     dport_dict[dport] = 'Port ' + str(le.inverse_transform([dport])[0])

    # le_agg_dport = le_agg_dport.reshape(len(le_agg_dport), 1)
    # print(dport_dict)
    # ohe_agg_dport = ohe.fit_transform(le_agg_dport)
    # ohe_agg_dport = pd.DataFrame(ohe_agg_dport).rename(columns=dport_dict)

    data = data.drop('Dst Port', axis=1)
    data = data.join(le_agg_dport)

    return data, labels

def cumulatively_categorise(column,threshold=0.90,return_categories_list=True):
    #Find the threshold value using the percentage and number of instances in the column
    threshold_value=int(threshold*len(column))
    #Initialise an empty list for our new minimised categories
    categories_list=[]
    #Initialise a variable to calculate the sum of frequencies
    s=0
    #Create a counter dictionary of the form unique_value: frequency
    counts=Counter(column)

    #Loop through the category name and its corresponding frequency after sorting the categories by descending order of frequency
    for i,j in counts.most_common():
        #Add the frequency to the global sum
        s+=dict(counts)[i]
        #Append the category name to the list
        categories_list.append(i)
        #Check if the global sum has reached the threshold value, if so break the loop
        if s>=threshold_value:
            break
    #Append the category Other to the list
    categories_list.append('Other')

    #Replace all instances not in our new categories by Other  
    new_column=column.apply(lambda x: x if x in categories_list else 'Other')

    #Return transformed column and unique values if return_categories=True
    if(return_categories_list):
        return new_column,categories_list
    #Return only the transformed column if return_categories=False
    else:
        return new_column

def replace_invalid(data, labels):
    """
    Cleans the data array.  The effect is to remove NaN and Inf values by using a nearest neighbor approach.
    Data deemed to be invalid will also be adjusted to the nearest valid value
    :param data: The data array
    :param labels: List of the labels for each sample from the data array
    :return: the processed data array
    """

    # Replace invalid values with average value of the column within each class label
    num_invalid = 0

    unique_labels = []
    for label in labels:
        if label not in unique_labels:
            unique_labels.append(label)

    class_avg = {}
    for label in unique_labels:
        index = np.full(len(labels), False)
        for i in range(len(labels)):
            if labels[i] == label:
                index[i] = True
        class_data = data[index, :]
        class_avg[label] = np.average(np.ma.masked_invalid(class_data), axis=0)

    for flow_idx in tqdm(range(data.shape[0]), file=sys.stdout, desc='Cleaning data array...'):
        label = labels[flow_idx]
        for attribute_idx in range(data.shape[1]):
            data_val = data[flow_idx, attribute_idx]
            if np.isnan(data_val) or np.isinf(data_val):
                data[flow_idx, attribute_idx] = class_avg[label][attribute_idx]
                num_invalid += 1
    print('Updated %d invalid values' % num_invalid)

    return data, labels, num_invalid

def resample_data(dset, data, labels):
    class_samples = {}
    orig_samples = len(labels)
    for label in labels:
        if label not in class_samples:
            class_samples[label] = 1
        else:
            class_samples[label] += 1
    save_class_hist(class_samples, 'orig_dist_' + dset)

    # Undersample Benign Data to 2x greater than next largest class
    benign_num = class_samples['Benign']
    largest_min_num = 0
    for class_name in class_samples.keys():
        if class_name != 'Benign' and class_samples[class_name] > largest_min_num:
            largest_min_num = class_samples[class_name]
    target_benign = 2 * largest_min_num
    if target_benign < benign_num:
        print('Reducing Benign data from %d to %d samples' % (benign_num, target_benign))
        undersampler = RandomUnderSampler(sampling_strategy={'Benign': target_benign})
        data, labels = undersampler.fit_resample(data, labels)
    else:
        print('Not reducing benign samples')

    print('Finished Undersampling')
    class_samples = {}
    for label in labels:
        if label not in class_samples:
            class_samples[label] = 1
        else:
            class_samples[label] += 1
    save_class_hist(class_samples, 'after_undersampling_' + dset)

    # # Drop extreme minority classes
    # min_class_count = 0.01 * class_samples['Benign']
    # print('Dropping classes with < %d samples' % min_class_count)

    # classes_to_drop = []
    # for class_name in class_samples.keys():
    #     if class_samples[class_name] < min_class_count:
    #         classes_to_drop.append(class_name)

    # data, labels = drop_classes(data, labels, classes_to_drop)

    # class_samples = {}
    # for label in labels:
    #     if label not in class_samples:
    #         class_samples[label] = 1
    #     else:
    #         class_samples[label] += 1
    # save_class_hist(class_samples, 'after_dropping_' + name)

    # Goal is to have all classes represented as 20% of benign data
    target_dict = {}
    target_num = round(class_samples['Benign'] * 0.20)
    print('Targeting %d samples for each minority class' % target_num)
    for label in class_samples.keys():
        if label == 'Benign' or class_samples[label] > target_num:
            target_dict[label] = class_samples[label]
        else:
            target_dict[label] = target_num

    oversampler = RandomOverSampler(sampling_strategy=target_dict)
    start = time.time()
    data, labels = oversampler.fit_resample(data, labels)
    print('Finished Oversampling')
    class_samples = {}
    for label in labels:
        if label not in class_samples:
            class_samples[label] = 1
        else:
            class_samples[label] += 1
    save_class_hist(class_samples, 'after_oversampling_' + dset)
    print('Total Data values: %d' % orig_samples)

    return data, labels

def drop_classes(data, labels, classes_to_drop):
    """
    Drops the classes specified in the classes_to_drop list from the data and labels structures
    :param data: The entire numpy dataset
    :param labels: The labels list for the data array
    :param classes_to_drop: A list of the classes that should be dropped from the dataset
    :return: The updated data and labels objects
    """
    drop_row = np.full(len(labels), False)
    for i in range(len(labels)):
        if labels[i] in classes_to_drop:
            drop_row[i] = True

    new_labels = []
    for i in range(len(labels)):
        if labels[i] not in classes_to_drop:
            new_labels.append(labels[i])
    labels = new_labels

    data = data[~drop_row, :]

    print('Done dropping.  Shape of data %s -- Size of labels %d' % (str(data.shape), len(labels)))
    return data, labels


def save_class_hist(samples_dict: dict, name: str):
    """
    Saves the histogram for the specified distribution.  Useful for comparing class distribution during preprocessing
    :param samples_dict: The dictionary containing class name as the key and the number of samples as the value
    :param name: Unique name for savefile name.
    :return: None
    """
    classes = samples_dict.keys()
    samples = []
    for class_name in classes:
        samples.append(samples_dict[class_name])

    plt.clf()
    plt.bar(classes, samples)
    plt.title('Class Distribution')
    plt.ylabel('Num Samples')
    plt.xlabel('Class Name')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('./out/', '%s.png' % name))  # TODO: Update to use specified output directory
    plt.clf()

def normalize(data):
    """
    Will normalize each column of a numpy array between 0-1 using the min-max method
    :param array: The data array
    :return: the normalized data
    """

    min = np.amin(data, axis=0)
    max = np.amax(data, axis=0)
    data -= min
    data /= (max - min + 1e-3)
    return data

def normalize_percentile(data):
    low = np.percentile(data, 5, axis=0)
    high = np.percentile(data, 95, axis=0)
    data -= low
    data /= (high - low + 1e-3)
    return data