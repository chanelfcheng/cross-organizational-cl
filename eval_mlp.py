import argparse
import math
import os
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from torch.utils.data import RandomSampler
from tqdm import tqdm

import mlp
from load_data import load_pytorch_datasets, CIC_2018, USB_2021

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help='Path to the pretrained weights')
    parser.add_argument('--data-path', type=str, required=True, help='Path to the dataset files')
    parser.add_argument('--dset', required=True, choices=[CIC_2017, CIC_2018], help='Specify which dataset to use for'
                                                                                    'evaluation')
    parser.add_argument('--batch-size', type=int, required=True, help='The batch size to use for evaluation')
    parser.add_argument('--name', type=str, default='debug', help='Unique name used for saving output files')
    parser.add_argument('--pkl-path', type=str, help='Path to store pickle files.  Saves time by storing preprocessed '
                                                     'data')
    parser.add_argument('--tsne', action='store_true', help='If set generates TSNE plots using subset of data.'
                                                            'Other metrics are not valid')
    parser.add_argument('--tsne-percent', default=0.01, help='To speed up TSNE, only run on a small portion of the '
                                                             'dataset')
    args = parser.parse_args()

    path = args.model_path
    if not os.path.exists(path):
        print('Path is invalid', file=sys.stderr)
        exit(1)

    eval_setup(path, args)
    print('Done')


if __name__ == '__main__':
    main()