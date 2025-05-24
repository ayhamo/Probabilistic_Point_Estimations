from configs.logger_config import global_logger as logger
from configs.config import DATASETS, RANDOM_STATE   

import os
import numpy as np
import pandas as pd

import openml

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import Pipeline

import torch
from torch.utils.data import DataLoader, TensorDataset

def create_loader(x_data, y_data, batch_size,shuffle = False):
    """
    Converts NumPy arrays into PyTorch tensors and creates a DataLoader with no shuffle.
    """
    x_tensor = torch.as_tensor(x_data, dtype=torch.float32)
    y_tensor = torch.as_tensor(y_data, dtype=torch.float32)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def load_preprocessed_data(model, source, dataset_identifier, fold = 0,
                           batch_size=2048, openml_pre_prcoess = False):

    # Error handling
    X_train, X_val, X_test, y_train, y_val, y_test = None, None, None, None, None, None
    
    dataset_name = DATASETS[source].get(str(dataset_identifier)).get("name")
    
    if source == 'uci':       
        logger.info(f"fetching {dataset_name}[fold {fold}], ({dataset_identifier}) locally.")

        # File Paths
        if os.path.exists("/kaggle/working"): # using kaggle then
            dataset_path = "/kaggle/working/Probabilistic_Point_Estimations/downloaded_datasets/UCI"
        else:
            dataset_path = "downloaded_datasets/UCI"

        current_dataset_path = os.path.join(dataset_path, dataset_identifier)

        fp_data = os.path.join(current_dataset_path, "data.txt")
        fp_index_features = os.path.join(current_dataset_path, "index_features.txt")
        fp_index_target = os.path.join(current_dataset_path, "index_target.txt")
        fp_index_train_rows = os.path.join(current_dataset_path, f"index_train_{fold}.txt")
        fp_index_test_rows = os.path.join(current_dataset_path, f"index_test_{fold}.txt")

        # Basic check for file existence
        required_files = [fp_data, fp_index_features, fp_index_target, fp_index_train_rows, fp_index_test_rows]
        for f_path in required_files:
            if not os.path.exists(f_path):
                logger.error(f"Required file not found for {dataset_identifier}: {f_path}")
                raise FileNotFoundError(f"Required file not found for {dataset_identifier}: {f_path}")

        # These will be pandas DataFrames
        x_train_full_raw_df = load_uci_data_segment(fp_data, fp_index_features, fp_index_train_rows)
        y_train_full_raw_df = load_uci_data_segment(fp_data, fp_index_target, fp_index_train_rows)
        x_test_raw_df = load_uci_data_segment(fp_data, fp_index_features, fp_index_test_rows)
        y_test_raw_df = load_uci_data_segment(fp_data, fp_index_target, fp_index_test_rows)
        
        # Convert to NumPy arrays
        x_train_full_np = x_train_full_raw_df.to_numpy()
        y_train_full_np = y_train_full_raw_df.to_numpy().ravel()
        x_test_np = x_test_raw_df.to_numpy()
        y_test_np = y_test_raw_df.to_numpy().ravel()
        
        if model == "TabPFN": # TODO i will test 5000 samples instead of 10000 samples
            X_train, y_train, X_test, y_test = reduce_dataset_size(x_train_full_np, y_train_full_np, x_test_np, y_test_np, max_samples=5000, random_state=RANDOM_STATE)

        if model != "TabResFlow":
            X_train, y_train, X_test, y_test = x_train_full_np, y_train_full_np, x_test_np, y_test_np
            # for other models than TabResFlow
            return X_train, y_train, X_test, y_test
        
        else:
            # Only for TabResFlow, it needs a loader + target scaling
            # Only for TabResFlow way:
            logger.info(f"pre-processing {dataset_identifier} with feature/taget scaling of [-1,1].")

            # Reshape y arrays if they are 1D to be 2D (N, 1) for scalers
            if y_train_full_np.ndim == 1:
                y_train_full_np = y_train_full_np.reshape(-1, 1)
            if y_test_np.ndim == 1:
                y_test_np = y_test_np.reshape(-1, 1)

            # Fit Scalers and transform
            feature_scaler = Pipeline(
                [("quantile", QuantileTransformer(output_distribution="normal")),
                ("standarize", StandardScaler()),])

            target_scaler = MinMaxScaler(feature_range=(-1, 1))

            # the code fit on training first
            feature_scaler.fit(x_train_full_np)
            target_scaler.fit(y_train_full_np)

            x_processed = feature_scaler.transform(x_train_full_np)
            y_processed = target_scaler.transform(y_train_full_np)

            x_test_processed = feature_scaler.transform(x_test_np)
            y_test_processed = target_scaler.transform(y_test_np)

            # make Validation Split from training
            # ResFlowDataModule(NodeFlow) split X_train into x_tr and x_val, with training of (0.8), so 1.0 - 0.8 = 0.2.
            x_tr_np, x_val_np, y_tr_np, y_val_np = train_test_split(
                x_processed, y_processed,
                test_size=0.2,
                random_state= RANDOM_STATE,
                shuffle=False
            )
            logger.info(f"Data split for {dataset_identifier}: "
                        f"Train X: {x_tr_np.shape}, Train Y: {y_tr_np.shape}, "
                        f"Validation X: {x_val_np.shape}, Validation Y: {y_val_np.shape}")
            
            # PyTorch TensorDatasets and DataLoaders
            # Convert processed NumPy arrays to PyTorch Tensors
            train_loader = create_loader(x_tr_np, y_tr_np, batch_size)
            val_loader = create_loader(x_val_np, y_val_np, batch_size)
            test_loader = create_loader(x_test_processed, y_test_processed, batch_size)

            logger.info(f"PyTorch DataLoaders created for UCI dataset {dataset_identifier}.")

            return train_loader, val_loader, test_loader, target_scaler
    
    # currently leaving multi to the end
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
            #X,y = load_naval_data(multi=True)
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
        logger.info(f"fetching {dataset_name}[fold {fold}] ({dataset_identifier}) from openML.")
        
        task = openml.tasks.get_task(int(dataset_identifier))
        dataset = task.get_dataset()
        X, y, categorical_indicator, attribute_names = dataset.get_data(target=task.target_name)  
        train_indices, test_indices = task.get_train_test_split_indices(fold=fold)

        if openml_pre_prcoess:

            if dataset_identifier == "361618":
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
            for col in categorical_cols:
                if X[col].isnull().any():
                    logger.info(f"Imputing missing values in categorical column '{col}' with '_MISSING_' for dataset {dataset_name}")

                    # Ensure '_MISSING_' is in the categories
                    if pd.api.types.is_categorical_dtype(X[col]):
                        X[col] = X[col].cat.add_categories("_MISSING_")  # Add '_MISSING_' as a valid category

                    # Fill missing values with '_MISSING_'
                    X[col] = X[col].fillna('_MISSING_') # Out-of-range imputation

            # 3. Impute missing numerical features
            from sklearn.impute import SimpleImputer
            if X[numerical_cols].isnull().any().any():
                logger.info(f"Imputing missing values in numerical columns for dataset {dataset_name} using mean.")
                num_imputer = SimpleImputer(strategy='mean')
                X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])

                X = pd.DataFrame(X, columns=attribute_names) # Restore dataframe structure

            # most important one as per the paper!
            # 4. One-hot encode categorical features
            if categorical_cols:
                logger.info(f"One-hot encoding categorical features for dataset {dataset_name}: {categorical_cols}")
                X = pd.get_dummies(X, columns=categorical_cols, prefix=categorical_cols, dummy_na=False) # dummy_na=False as we imputed 
                # Convert one-hot encoded bool columns to int (0/1), as pytorch dataset can only handle numerical
                bool_cols_after_encoding = X.select_dtypes(include=['bool']).columns  # Identify new bool columns
                X[bool_cols_after_encoding] = X[bool_cols_after_encoding].astype(int)  # Convert to integer

                
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_np = y.to_numpy()
        else:
            y_np = y

        # X should already be a DataFrame after pd.get_dummies. Convert to NumPy.
        x_np_full = X.to_numpy()

        # split given indicies from openml task
        X_train = x_np_full[train_indices]
        y_train = y_np[train_indices]
        X_test = x_np_full[test_indices]
        y_test = y_np[test_indices]

        # If there are more than 10,000 samples, randomly sample 10,000 indices, 
        # that's becuase TabPFN does not work with more than that.
        if model == "TabPFN": # TODO i will test 5000 samples instead of 10000 samples
            X_train, y_train, X_test, y_test = reduce_dataset_size(X_train, y_train, X_test, y_test, max_samples=5000, random_state=RANDOM_STATE)

        if model != "TabResFlow":
            # for all models
            return X_train, y_train, X_test, y_test

        else:
                
            logger.info(f"pre-processing {dataset_identifier} with feature/taget scaling of [-1,1].")
        
            # Reshape y arrays if they are 1D to be 2D (N, 1) for scalers
            if y_train.ndim == 1:
                y_train = y_train.reshape(-1, 1)
            if y_test.ndim == 1:
                y_test = y_test.reshape(-1, 1)

            # Fit Scalers and transform
            feature_scaler = Pipeline(
                [("quantile", QuantileTransformer(output_distribution="normal")),
                ("standarize", StandardScaler()),])

            target_scaler = MinMaxScaler(feature_range=(-1, 1))

            # the code fit on training first
            feature_scaler.fit(X_train)
            target_scaler.fit(y_train)

            x_processed = feature_scaler.transform(X_train)
            y_processed = target_scaler.transform(y_train)

            x_test_processed = feature_scaler.transform(X_test)
            y_test_processed = target_scaler.transform(y_test)

            # make Validation Split from training
            # ResFlowDataModule(NodeFlow) split X_train into x_tr and x_val, with training of (0.8), so 1.0 - 0.8 = 0.2.
            x_tr_np, x_val_np, y_tr_np, y_val_np = train_test_split(
                x_processed, y_processed,
                test_size=0.2,
                random_state= RANDOM_STATE,
                shuffle=False
            )
            logger.info(f"Data split for {dataset_identifier}: "
                        f"Train X: {x_tr_np.shape}, Train Y: {y_tr_np.shape}, "
                        f"Validation X: {x_val_np.shape}, Validation Y: {y_val_np.shape}")
            
            # PyTorch TensorDatasets and DataLoaders
            # Convert processed NumPy arrays to PyTorch Tensors
            train_loader = create_loader(x_tr_np, y_tr_np, batch_size)
            val_loader = create_loader(x_val_np, y_val_np, batch_size)
            test_loader = create_loader(x_test_processed, y_test_processed, batch_size)

            logger.info(f"PyTorch DataLoaders created for OpenML-CTR23 dataset {dataset_identifier}.")

            return train_loader, val_loader, test_loader, target_scaler


def load_uci_data_segment(filepath_data,
                          
                          filepath_index_columns,
                          filepath_index_rows,
                          data_delimiter=None,
                          index_columns_delimiter=None,
                          index_rows_delimiter=None):
    """
    Loads a segment of UCI data based on data file and index files for rows and columns.
    Mimics the behavior of the provided UCIDataSet._load method.
    """
    # Load the entire data matrix
    data_full = np.loadtxt(filepath_data, delimiter=data_delimiter)
    df_full = pd.DataFrame(data_full)

    # Load column indices and reshape to be 1D
    index_columns = np.loadtxt(filepath_index_columns, dtype=np.int32, delimiter=index_columns_delimiter)
    index_columns = index_columns.reshape(-1)

    # Load row indices and reshape to be 1D
    index_rows = np.loadtxt(filepath_index_rows, dtype=np.int32, delimiter=index_rows_delimiter)
    index_rows = index_rows.reshape(-1)

    # Select the specified rows and columns
    return df_full.iloc[index_rows, index_columns]

def reduce_dataset_size(X_train, y_train, X_test, y_test, max_samples=10000, random_state=None):
    """
    Reduces the dataset size to max_samples if it exceeds the limit.
    
    Parameters:
    X_train, y_train, X_test, y_test: NumPy arrays representing training and test datasets.
    max_samples: Maximum number of samples allowed.
    random_state: Seed for reproducibility.
    
    Returns:
    Reduced datasets (X_train, y_train, X_test, y_test).
    """
    rng = np.random.RandomState(random_state)

    if X_train.shape[0] > max_samples:
        logger.warning(f"Dataset has more than {max_samples} samples, reducing to {max_samples} samples for TabPFN to work")
        subsample_indices_train = rng.choice(X_train.shape[0], size=max_samples, replace=False)
        X_train = X_train[subsample_indices_train]
        y_train = y_train[subsample_indices_train]

    if X_test.shape[0] > max_samples:
        subsample_indices_test = rng.choice(X_test.shape[0], size=max_samples, replace=False)
        X_test = X_test[subsample_indices_test]
        y_test = y_test[subsample_indices_test]

    return X_train, y_train, X_test, y_test

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
