import pandas as pd

# Load in sample CIC 2018 dataset
sample_cic_file = '/home/chanel/Cyber/yang-summer-2022/data/CIC-IDS2018/02-14-2018.csv'
sample_cic_df = pd.read_csv(sample_cic_file, dtype=str, skipinitialspace=True)

# Remove unused columns if present
if 'Timestamp' in sample_cic_df:
    sample_cic_df = sample_cic_df.drop('Timestamp', axis=1)
if 'Flow ID' in sample_cic_df:
    sample_cic_df = sample_cic_df.drop('Flow ID', axis=1)
if 'Src IP' in sample_cic_df:
    sample_cic_df = sample_cic_df.drop('Src IP', axis=1)
if 'Src Port' in sample_cic_df:
    sample_cic_df = sample_cic_df.drop('Src Port', axis=1)
if 'Dst IP' in sample_cic_df:
    sample_cic_df = sample_cic_df.drop('Dst IP', axis=1)

# Get features
cic_2018_features = sample_cic_df.columns.values.tolist()

# Load in sample USB 2021 dataset
sample_usb_file = '/home/chanel/Cyber/yang-summer-2022/data/USB-IDS2021/Hulk-Evasive.csv'
sample_usb_df = pd.read_csv(sample_usb_file, dtype=str, skipinitialspace=True)

# Remove unused features
if 'Timestamp' in sample_usb_df:
    sample_usb_df = sample_usb_df.drop('Timestamp', axis=1)
if 'Flow ID' in sample_usb_df:
    sample_usb_df = sample_usb_df.drop('Flow ID', axis=1)
if 'Src IP' in sample_usb_df:
    sample_usb_df = sample_usb_df.drop('Src IP', axis=1)
if 'Src Port' in sample_usb_df:
    sample_usb_df = sample_usb_df.drop('Src Port', axis=1)
if 'Dst IP' in sample_usb_df:
    sample_usb_df = sample_usb_df.drop('Dst IP', axis=1)

# Get features
usb_2021_features = sample_usb_df.columns.values.tolist()

def get_feature_map():
    feature_map = {}
    for i in range(len(usb_2021_features)):
        feature_map[usb_2021_features[i]] = cic_2018_features[i]
    return feature_map

def main():
    feature_map = get_feature_map()
    print(feature_map)

if __name__ == '__main__':
    main()