from collections import Counter
import logging as logger
import os
import shutil

import numpy as np
import pandas as pd
import urllib


def download_multivariate_datasets():
    url = "https://zenodo.org/record/1161203/files/data.tar.gz?download=1"
    dataset_dir = 'downloaded_datasets'
    data_dir = os.path.join(dataset_dir, "data")  # extracted data folder
    archive_path = os.path.join(dataset_dir, "data.tar.gz")

    os.makedirs(dataset_dir, exist_ok=True)

    # Check if data folder already exists
    if not os.path.exists(data_dir):
        logger.info(f"Downloading multivariate UCI datasets from: {url}...")
        urllib.request.urlretrieve(url, archive_path)
        
        # Extract archive
        shutil.unpack_archive(archive_path, dataset_dir)
        logger.info(f"Multivariate dataset extracted to {dataset_dir}.")
        os.remove(archive_path)
        
        # Remove unwanted folders
        unwanted_folders = ["BSDS300", "cifar10", "mnist"]
        for folder in unwanted_folders:
            folder_path = os.path.join(data_dir, folder)
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)

    else:
        logger.info(f"Data folder already exists, skipping download.")

def process_miniboone_data():
    # Adapted from: https://github.com/gpapamak/maf/blob/master/datasets/miniboone.py
    # Dataset description: https://archive.ics.uci.edu/dataset/199/miniboone+particle+identification

    #TODO ask about here, no regression, just calssifacation, can do regression on a continus sensor data?
    data = np.load("downloaded_datasets/data/miniboone/data.npy")

    # Extract target (first column)
    y = data[:, 0].reshape(-1, 1)  # First column as target

    # Extract features (remaining columns)
    X = data[:, 1:]

    # Remove features with too many repeated values
    features_to_remove = [i for i, feature in enumerate(X.T) if Counter(feature).most_common(1)[0][1] > 5]
    X = np.delete(X, features_to_remove, axis=1)

    # Normalize data
    mu = X.mean(axis=0)
    s = X.std(axis=0)
    X = (X - mu) / s
    y = (y - y.mean()) / y.std()

    # Split dataset
    N_test = int(0.1 * X.shape[0])
    X_test, y_test = X[-N_test:], y[-N_test:]
    X_train, y_train = X[:-N_test], y[:-N_test]

    N_validate = int(0.1 * X_train.shape[0])
    X_val, y_val = X_train[-N_validate:], y_train[-N_validate:]
    X_train, y_train = X_train[:-N_validate], y_train[:-N_validate]

    return X_train, y_train, X_val, y_val, X_test, y_test



def load_hepmass_data(old=False):
    # Adapted from: https://github.com/gpapamak/maf/blob/master/datasets/hepmass.py
    # Dataset description: https://archive.ics.uci.edu/dataset/347/hepmass

    #TODO ask about here, no regression, just calssifacation, can do regression on a continus sensor data?
    # other wise i could make naval also multi
    if old:
        # Load original CSV files
        data_train = pd.read_csv("downloaded_datasets/HEPMASS_train.csv", index_col=False)
        data_test = pd.read_csv("downloaded_datasets/HEPMASS_test.csv", index_col=False)

        # Remove background noise (class 0)
        data_train = data_train[data_train.iloc[:, 0] == 1].drop(columns=[data_train.columns[0]])
        data_test = data_test[data_test.iloc[:, 0] == 1].drop(columns=[data_test.columns[0]])

        # Remove last column from test data (dataset issue)
        data_test = data_test.iloc[:, :-1]
    else:
        # Load preprocessed NumPy file
        data = np.load("downloaded_datasets/data/hepmass.npy")
        data = pd.DataFrame(data)  # Convert NumPy array to Pandas DataFrame

    # Extract target (`y`) and features (`X`)
    y = data.iloc[:, 0].to_frame()  # First column as target
    X = data.iloc[:, 1:]  # Remaining columns as features

    # Remove features with too many repeated values
    features_to_remove = [i for i, feature in enumerate(X.T.values) if Counter(feature).most_common(1)[0][1] > 5]
    X.drop(X.columns[features_to_remove], axis=1, inplace=True)

    # Normalize data
    X = (X - X.mean()) / X.std()
    y = (y - y.mean()) / y.std()

    # Split dataset
    N_test = int(0.1 * X.shape[0])
    X_test, y_test = X.iloc[-N_test:], y.iloc[-N_test:]
    X_train, y_train = X.iloc[:-N_test], y.iloc[:-N_test]

    N_validate = int(0.1 * X_train.shape[0])
    X_val, y_val = X_train.iloc[-N_validate:], y_train.iloc[-N_validate:]
    X_train, y_train = X_train.iloc[:-N_validate], y_train.iloc[:-N_validate]

    return X_train, y_train, X_val, y_val, X_test, y_test