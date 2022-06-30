import glob
import pandas as pd
import numpy as np
from load_data import load_datasets
from utils.data_preprocessing import process_features, replace_invalid, resample_data, CIC_2018, USB_2021
from sklearn.preprocessing import MinMaxScaler, RobustScaler

def main():
    for file in list(glob.glob('/home/chanel/Cyber/yang-summer-2022/data/CIC-IDS2018/Hulk-Slowloris/*.csv')):
        print('Loading ', file, '...')
        reader = pd.read_csv(file, dtype=str, chunksize=10**6, skipinitialspace=True)  # Read in data from csv file

        for df in reader:
            data, labels = process_features(CIC_2018, df.sample(n=10000))

            # Convert dataframe to numpy array for processing
            data_np = np.array(data.to_numpy(), dtype=float)
            labels_lst = labels.tolist()

            data_np, labels_lst, num_invalid = replace_invalid(data_np, labels_lst)  # Clean data
            
            scaler = RobustScaler(quantile_range=(5,95))
            data_scale = scaler.fit_transform(data_np)
            print(np.percentile(data_scale, 96, axis=0))
            break
        break

if __name__ == '__main__':
    main()