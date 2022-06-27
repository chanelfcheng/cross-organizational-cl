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

# Get attributes
cic_2018_attributes = sample_cic_df.columns.values.tolist()

# Load in sample USB 2021 dataset
sample_usb_file = '/home/chanel/Cyber/yang-summer-2022/data/USB-IDS2021/USB-IDS-1-TEST.csv'
sample_usb_df = pd.read_csv(sample_usb_file, dtype=str, skipinitialspace=True)

# Remove unused attributes
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

# Get attributes
usb_2021_attributes = sample_usb_df.columns.values.tolist()

def get_attribute_map():
    attribute_map = {}
    for i in range(len(usb_2021_attributes)):
        attribute_map[usb_2021_attributes[i]] = cic_2018_attributes[i]
    return attribute_map

def main():
    attribute_map = get_attribute_map()
    print(attribute_map)

if __name__ == '__main__':
    main()