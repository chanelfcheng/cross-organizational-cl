import argparse
import os
import time

from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from load_data import load_datasets
from utils.data_preprocessing import CIC_2018, USB_2021


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-depth', type=int, default=10, help='Max Depth Hyperparam for RF')
    parser.add_argument('--data-path', type=str, required=True, help='Path to root directory of dataset')
    parser.add_argument('--dset', type=str, required=True, choices=[CIC_2018, USB_2021], help='Dataset to classify')
    parser.add_argument('--pkl-path', type=str, default=None,  help='Path to stored pickle files.  Saves time by '
                                                                    'loading preprocessed data')

    args = parser.parse_args()

    clf = RandomForestClassifier(max_depth=args.max_depth, random_state=0, n_jobs=20, verbose=1)
    print(args.dset)
    print(args.data_path)
    print(args.pkl_path)
    data_train, data_test, labels_train, labels_test = load_datasets(name=args.dset, data_path=args.data_path, pkl_path=args.pkl_path)

    print('\n\n-----------------------------------------------------------\n')
    print('Fitting RF Model')
    start = time.time()
    clf.fit(data_train, labels_train)
    print('Training took %.2f minutes' % ((time.time() - start) / 60.0))

    predictions = clf.predict(data_test)

    print(classification_report(labels_test, predictions))
    cf_matrix = confusion_matrix(labels_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix)
    disp.plot()
    plt.savefig(os.path.join('./out/', 'rf_cf.png'))

    print('Done')


if __name__ == '__main__':
    main()