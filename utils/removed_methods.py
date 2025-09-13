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

def load_power_grid_data():
    # Adapated from: https://github.com/gpapamak/maf/blob/master/datasets/power.py
    # Dataset description: http://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption

    rng = np.random.RandomState(42)
    # data is Global_active_power, Global_reactive_power, Voltage, Global_intensity, 
    # Sub_metering_1, Sub_metering_2, Sub_metering_3, Derived time feature (maybe a time stamp? not in original)
    data = np.load("downloaded_datasets/data/power/data.npy")

    logger.info(f"Processing multivariate dataset power from UCI")
    rng.shuffle(data)
    N = data.shape[0]

    # global_intensity, instead of dropping i set it also as target with global_active_power for 2 multi target
    #data = np.delete(data, 3, axis=1)
    # global_reactive_power
    data = np.delete(data, 1, axis=1)

    # Add noise.
    global_intensity_noise = 0.1*rng.rand(N, 1)
    voltage_noise = 0.01 * rng.rand(N, 1)
    gap_noise = 0.001 * rng.rand(N, 1)
    sm_noise = rng.rand(N, 3)
    time_noise = np.zeros((N, 1))
    noise = np.hstack((gap_noise, voltage_noise, global_intensity_noise, sm_noise, time_noise))
    data = data + noise

    y = data[:, :2]  # First column global active power as target
    X = data[:, 2:]  # Remaining columns as features

    # Split dataset into train, validation, and test sets
    N_test = int(0.1 * X.shape[0])
    X_test, y_test = X[-N_test:], y[-N_test:]
    X, y = X[:-N_test], y[:-N_test]

    N_validate = int(0.1 * X.shape[0])
    X_val, y_val = X[-N_validate:], y[-N_validate:]
    X_train, y_train = X[:-N_validate], y[:-N_validate]

    # Normalize data using training + validation statistics
    mu = np.concatenate((X_train, X_val)).mean(axis=0)
    s = np.concatenate((X_train, X_val)).std(axis=0)
    X_train = (X_train - mu) / s
    X_val = (X_val - mu) / s
    X_test = (X_test - mu) / s

    return X_train, X_test, X_val, y_train, y_test, y_val

def get_correlation_numbers(data):
    C = data.corr()
    A = C > 0.98
    B = A.values.sum(axis=1)
    return B


def load_gas_data(old=False):
    # Adapted from: https://github.com/gpapamak/maf/blob/master/datasets/gas.py
    # Dataset description: http://archive.ics.uci.edu/ml/datasets/Gas+sensor+array+under+dynamic+gas+mixtures

    if old:
        # this is broken due to pandas changes, please check https://github.com/gpapamak/maf/tree/master?tab=readme-ov-file#how-to-get-the-datasets
        data = pd.read_pickle("data/gas/ethylene_CO.pickle")
        data.drop("Time", axis=1, inplace=True)
    else:
        data = np.load("downloaded_datasets/data/gas/gas.npy") 
        # Time CO2_conc_(ppm) Ethylene_conc_(ppm) Sensor1 ... Sensor16
        #TODO handle to download it from somehwere! currently can be gotten from https://www.kaggle.com/code/ayhamo/fix-multivarate-datasets
        data = pd.DataFrame(data)
        # Drop the first column (Time)
        data.drop(columns=[0], inplace=True)

   # target (second & third columns) - 2 feautres
    y = data.iloc[:, :2] 

    X = data.iloc[:, 2:]

    # Remove highly correlated columns
    B = get_correlation_numbers(X)  # only feautres
    while np.any(B > 1):
        col_to_remove = np.where(B > 1)[0][0]
        col_name = X.columns[col_to_remove]
        X = X.drop(col_name, axis=1)
        B = get_correlation_numbers(X)

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