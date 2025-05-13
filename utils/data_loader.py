from configs.logger_config import global_logger as logger
from configs.config import DATASETS, RANDOM_STATE   

import os
import numpy as np
import pandas as pd
import shutil
import urllib.request
from collections import Counter

from uci_datasets import Dataset
from ucimlrepo import fetch_ucirepo  # for uci power dataset
import openml

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader

# PyTorch Dataset
class TabularDataset(TorchDataset):
    def __init__(self, X_num_df, y_series, X_cat_df=pd.DataFrame()): # for now we do not have catorgical data, so empty
        self.X_num = torch.tensor(X_num_df.values, dtype=torch.float32)
        
        if X_cat_df.empty:
            self.X_cat = torch.empty((len(X_num_df), 0), dtype=torch.long)
        else:
            self.X_cat = torch.tensor(X_cat_df.values, dtype=torch.long)
            
        self.y = torch.tensor(y_series.values, dtype=torch.float32).unsqueeze(1) # Ensure y is [batch, 1]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], self.y[idx]

def load_preprocessed_data(source, dataset_identifier, batch_size=2048):

    # Error handling
    X_train, X_val, X_test, y_train, y_val, y_test = None, None, None, None, None, None
    categorical_indicator = None # For openml
    
    dataset_name = DATASETS[source].get(str(dataset_identifier)).get("name")
    if source == 'uci':

        if dataset_identifier == "naval":
            logger.info(f"fetching {dataset_name} ({dataset_identifier}) from uci as zip.")
            X,y = load_naval_data(multi=False)
        
        elif dataset_identifier == "power":
            logger.info(f"fetching {dataset_name} ({dataset_identifier}) from ucirepo.")
            # fetch dataset 
            combined_cycle_power_plant = fetch_ucirepo(id=294) 
            X = combined_cycle_power_plant.data.features 
            y = combined_cycle_power_plant.data.targets 
    
        elif dataset_identifier == "kin8nm":
            logger.info(f"fetching {dataset_name}, ID: 189 from OpenML.")
            dataset = openml.datasets.get_dataset(189)
            X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)  

                    
        else:
            logger.info(f"fetching {dataset_name} ({dataset_identifier}) from UCI.")
            data = Dataset(dataset_identifier)
            x_train, y_train, x_test, y_test = data.get_split(split=0)

            x_train = pd.DataFrame(x_train)
            x_test = pd.DataFrame(x_test)
            y_train = pd.DataFrame(y_train)
            y_test = pd.DataFrame(y_test)
            
            X = pd.concat([x_train, x_test], axis=0)
            y = pd.concat([y_train, y_test], axis=0)
        
        logger.info(f"Processing {dataset_name}...")

        # then pre-prcoess
        # Kiran paper and NodeFlow scaling both input features and target variables to the range [-1, 1]
        #   Kiran paper uses MLP to handle numerical/catorgial features
        df = pd.concat([X, y], axis=1)
        numeric_cols = df.select_dtypes(include=np.number).columns

        final_df = df.copy()

        # non numreic columns are not touched
        if len(numeric_cols) > 0:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            final_df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        final_df = final_df[df.columns] # ensure order

        X_scaled = final_df.iloc[:, :-1]  # All columns except last
        y_scaled = final_df.iloc[:, -1]   # Last column

        # Split scaled data into training, validation, and test sets (80% train, 20% test)
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.10, random_state=RANDOM_STATE, shuffle=False)
        
        # Second split: train and validation
        # test_size=0.20 means 20% of X_train_val (which is 80% of total) goes to validation.
        # 0.2 * 0.9 = 0.18 (18% of total for validation)
        # these are dervied from NodeFlow paper
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.20, random_state=RANDOM_STATE, shuffle=False)
    
    elif source == 'multivariate':
        logger.info(f"fetching {dataset_name} ({dataset_identifier}) from UCI.")
        
        if dataset_identifier == "gas":
            X_train, X_test, X_val, y_train, y_test, y_val = load_gas_data()
        
        if dataset_identifier == "power_grid":
            X_train, X_test, X_val, y_train, y_test, y_val = load_power_grid_data()
            
            # Convert NumPy arrays to Pandas DataFrames
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)
            X_val = pd.DataFrame(X_val)
            y_train = pd.DataFrame(y_train)
            y_test = pd.DataFrame(y_test)
            y_val = pd.DataFrame(y_val)

        if dataset_identifier == "naval_multi":
            X,y = load_naval_data(multi=True)
            # pre-prcoess same as above

            df = pd.concat([X, y], axis=1)
            numeric_cols = df.select_dtypes(include=np.number).columns

            final_df = df.copy()
            if len(numeric_cols) > 0:
                scaler = MinMaxScaler(feature_range=(-1, 1))
                final_df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

            final_df = final_df[df.columns]
            X_scaled = final_df.iloc[:, :-2]  # All columns except the last two
            y_scaled = final_df.iloc[:, -2:]  # Last two columns as target

            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X_scaled, y_scaled, test_size=0.10, random_state=RANDOM_STATE, shuffle=False)
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=0.20, random_state=RANDOM_STATE, shuffle=False)
    
    elif source == 'openml_ctr23':
        logger.info(f"fetching {dataset_name} ({dataset_identifier}) from openML.")
        dataset = openml.datasets.get_dataset(dataset_identifier)
        X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute)  
        
        # Manual fix for forest_fires dataset (ID 44962) with day and month, we one hot encode them as per paper.
        if dataset_identifier == "44962":
            categorical_indicator[2] = True  # month
            categorical_indicator[3] = True  # day


        categorical_cols = [col for col, is_cat in zip(X.columns, categorical_indicator) if is_cat]
        numerical_cols = [col for col in X.columns if col not in categorical_cols]
        
        # from Appendix B.2, code by ChatGPT
        # 1. Collapse rarest categorical levels
        # my selection has no such thing.
        MAX_CATEGORICAL_LEVELS = 1000 # From Appendix B.2
        for col in categorical_cols:
            if X[col].nunique() > MAX_CATEGORICAL_LEVELS:
                logger.info(f"Collapsing categories for column '{col}' in dataset: {dataset_name} (had {X[col].nunique()} levels)")
                top_categories = X[col].value_counts().nlargest(MAX_CATEGORICAL_LEVELS - 1).index
                X[col] = X[col].apply(lambda x: x if pd.isna(x) or x in top_categories else '_RARE_')

        # 2. Impute missing categorical values (using a constant placeholder like "_MISSING_")
        # my selection has no such thing
        for col in categorical_cols:
            if X[col].isnull().any():
                logger.info(f"Imputing missing values in categorical column '{col}' with '_MISSING_' for dataset {dataset_name}")
                X[col] = X[col].fillna('_MISSING_') # Out-of-range imputation

        # 3. Impute missing numerical features
        if X[numerical_cols].isnull().any().any():
            logger.info(f"Imputing missing values in numerical columns for dataset {dataset_name} using median.")
            num_imputer = SimpleImputer(strategy='mean')
            X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])
            X = pd.DataFrame(X, columns=attribute_names) # Restore dataframe structure

        # most important one as per the paper!
        # 4. One-hot encode categorical features
        if categorical_cols:
            logger.info(f"One-hot encoding categorical features for dataset {dataset_name}: {categorical_cols}")
            X = pd.get_dummies(X, columns=categorical_cols, prefix=categorical_cols, dummy_na=False) # dummy_na=False as we imputed 

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)  # 80% train, 20% temp
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # Split temp into 50% test, 50% validation
    

    # PyTorch Datasets and DataLoaders
    train_torch_dataset = TabularDataset(X_train, y_train)
    val_torch_dataset = TabularDataset(X_val, y_val)
    test_torch_dataset = TabularDataset(X_test, y_test)

    train_loader = DataLoader(train_torch_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_torch_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_torch_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader, X_train.shape[1] ,dataset_name


def load_naval_data(multi):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00316/UCI%20CBM%20Dataset.zip'
    
    dataset_dir = 'downloaded_datasets'
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Paths for zip file and extracted data
    zip_path = os.path.join(dataset_dir, "condition+based+maintenance+of+naval+propulsion+plants.zip")
    data_file_path = os.path.join(dataset_dir, 'UCI CBM Dataset', 'data.txt')
    
    if not os.path.isfile(data_file_path):
        logger.info(f"Downloading naval dataset datasets from: {url}...")
        urllib.request.urlretrieve(url, zip_path)
        
        # extract
        shutil.unpack_archive(zip_path, dataset_dir)
        logger.info(f"naval Dataset extracted to {dataset_dir}.")
        # then delete it
        os.remove(zip_path)

    # Read data using pandas.read_fwf
    data_df = pd.read_fwf(data_file_path, header=None)
    
    # there exist 2 targets, we pick 1 as per other papers, GT Turbine Decay State Coefficient (second-to-last column)
    # also adpated it for multi target with both of them
    if multi:
        X = data_df.iloc[:, :-2]  # All columns except the last two
        y = data_df.iloc[:, -2].to_frame()  # Second-to-last column as DataFrame
    else:
        y = data_df.iloc[:, -2].to_frame()  # Second-to-last column as target (shape: [n_samples, 1])
        X = data_df.drop(data_df.columns[-2], axis=1)  # All columns except the second-to-last
    
    return X, y


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
