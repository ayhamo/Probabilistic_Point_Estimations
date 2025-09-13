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

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

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

        if model == "ARMD":
            train_df = pd.concat([x_train_full_raw_df, y_train_full_raw_df], axis=1)
            test_df = pd.concat([x_test_raw_df, y_test_raw_df], axis=1)
            # Combine train and test into one DataFrame
            full_df = pd.concat([train_df, test_df], axis=0)

            # Shuffle the combined DataFrame
            full_df = full_df.sample(frac=1).reset_index(drop=True)
            data = full_df.values
            
            scaler = StandardScaler()
            scaler = scaler.fit(data)
            return data, scaler
        
        # Convert to NumPy arrays
        X_train = x_train_full_raw_df.to_numpy()
        y_train = y_train_full_raw_df.to_numpy().ravel()
        X_test = x_test_raw_df.to_numpy()
        y_test = y_test_raw_df.to_numpy().ravel()

        # return the pandas dataframe for TabResNet for X, and y as numpy due to evulation
        if model == "TabResNet":
            return x_train_full_raw_df, y_train, x_test_raw_df, y_test
        
        if model == "TabPFN":
            X_train, y_train, X_test, y_test = reduce_dataset_size(X_train, y_train, X_test, y_test, max_samples=10000, random_state=RANDOM_STATE)
        
        # This model reuqires raw data, train and test only!
        if model == "TTVAE":
            train_df = pd.concat([x_train_full_raw_df, y_train_full_raw_df], axis=1)
            test_df = pd.concat([x_test_raw_df, y_test_raw_df], axis=1)
            return train_df, test_df
        
        if model != "TabResFlow":
            # for other models than TabResFlow (TDGPs)
            return X_train, y_train, X_test, y_test
        
        else:

            x_train_full_np = X_train
            y_train_full_np = y_train
            x_test_np = X_test
            y_test_np = y_test

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
            X_train, X_test, X_val, y_train, y_test, y_val = None#load_gas_data()
        
        if dataset_identifier == "power_grid":
            X_train, X_test, X_val, y_train, y_test, y_val = None#load_power_grid_data()
            
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

        # Convert to DataFrame immediately for pre-processing
        X = pd.DataFrame(X, columns=attribute_names)
        y = y.values if hasattr(y, 'values') else y  # Ensure numpy array

        # split given indicies from openml task
        X_train_raw = X.iloc[train_indices]
        y_train = y[train_indices]
        X_test_raw = X.iloc[test_indices]
        y_test = y[test_indices]

        if openml_pre_prcoess:

            # Initialize preprocessor and fit ONLY on training data
            preprocessor = make_openml_preprocessor(X_train_raw, include_onehot=True)
            preprocessor.fit(X_train_raw)

            X_train = preprocessor.transform(X_train_raw)
            X_test = preprocessor.transform(X_test_raw)
        else:
            # TabResNet requires a pd frame, so i just return it here without coverting to numpy
            # and also it does not need one hot encoding, just imputer if needed, so 
            if model == "TabResNet":
                # we make the whole pre-processor pipeline without one-hot encoding, mainly imputing
                preprocessor = make_openml_preprocessor(X_train_raw, include_onehot=False)
                preprocessor.fit(X_train_raw)

                X_train = preprocessor.transform(X_train_raw)
                X_test = preprocessor.transform(X_test_raw)

                return pd.DataFrame(X_train, columns=X_train_raw.columns), y_train, \
                    pd.DataFrame(X_test, columns=X_test_raw.columns), y_test
            
            # not sure if this is being used?
            else:
                # else for all models, just convert to numpy
                X_train = X_train_raw.to_numpy()
                X_test = X_test_raw.to_numpy()

        # If there are more than 10,000 samples, randomly sample 10,000 indices, 
        # that's becuase TabPFN does not work with more than that.
        if model == "TabPFN":
            X_train, y_train, X_test, y_test = reduce_dataset_size(X_train, y_train, X_test, y_test, max_samples=10000, random_state=RANDOM_STATE)

        # This model reuqires raw data, train and test only!
        if model == "TTVAE":
            train_df = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train)], axis=1)
            test_df = pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test)], axis=1)
            return train_df, test_df

        if model == "ARMD":
            train_df = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train)], axis=1)
            test_df = pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test)], axis=1)
            # Combine train and test into one DataFrame
            full_df = pd.concat([train_df, test_df], axis=0)

            # Shuffle the combined DataFrame
            full_df = full_df.sample(frac=1).reset_index(drop=True)
            data = full_df.values
            
            scaler = StandardScaler()
            scaler = scaler.fit(data)
            return data, scaler

        if model != "TabResFlow":
            # for all models
            return X_train, y_train, X_test, y_test

        else:
            # since TabResFlow UCI had target scaling, i also do target scaling here, only valid for TabResFlow
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

from torch.utils.data import Dataset
from models.ARMDMain.Models.autoregressive_diffusion.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from models.ARMDMain.Utils.masking_utils import noise_mask


class ARMDDataset(Dataset):
    def __init__(
        self,
        source,
        dataset_identifier,
        fold,
        window=64, 
        proportion=0.8, 
        save2npy=False, 
        neg_one_to_one=True,
        seed=123,
        period='train',
        output_dir='./OUTPUT',
        predict_length=None,
        missing_ratio=None,
        style='separate', 
        distribution='geometric', 
        mean_mask_length=3
    ):
        super(ARMDDataset, self).__init__()
        assert period in ['train', 'test'], 'period must be train or test.'
        if period == 'train':
            assert not(predict_length is not None or missing_ratio is not None), ''

        self.name, self.pred_len, self.missing_ratio = dataset_identifier, predict_length, missing_ratio
        self.style, self.distribution, self.mean_mask_length = style, distribution, mean_mask_length
        #self.rawdata, self.scaler = self.read_data(data_root, self.name)
        self.rawdata, self.scaler = load_preprocessed_data("ARMD", source, dataset_identifier, fold,
                           batch_size=0, openml_pre_prcoess = True)
        self.dir = os.path.join(output_dir, 'samples')
        os.makedirs(self.dir, exist_ok=True)

        self.window, self.period = window, period
        self.len, self.var_num = self.rawdata.shape[0], self.rawdata.shape[-1]
        self.sample_num_total = max(self.len - self.window + 1, 0)
        self.save2npy = save2npy
        #self.auto_norm = neg_one_to_one
        self.auto_norm = False

        self.data = self.__normalize(self.rawdata)
        train, inference = self.__getsamples(self.data, proportion, seed)

        self.samples = train if period == 'train' else inference
        if period == 'test':
            if missing_ratio is not None:
                self.masking = self.mask_data(seed)
            elif predict_length is not None:
                masks = np.ones(self.samples.shape)
                masks[:, -predict_length:, :] = 0
                self.masking = masks.astype(bool)
            else:
                raise NotImplementedError()
        self.sample_num = self.samples.shape[0]
  

    def __getsamples(self, data, proportion, seed):
        x = np.zeros((self.sample_num_total, self.window, self.var_num))
        for i in range(self.sample_num_total):
            start = i
            end = i + self.window
            x[i, :, :] = data[start:end, :]

        train_data, test_data = self.divide(x, proportion, seed)

        if self.save2npy:
            if 1 - proportion > 0:
                np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_test.npy"), self.unnormalize(test_data))
            np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_train.npy"), self.unnormalize(train_data))
            if self.auto_norm:
                if 1 - proportion > 0:
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), unnormalize_to_zero_to_one(test_data))
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), unnormalize_to_zero_to_one(train_data))
            else:
                if 1 - proportion > 0:
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), test_data)
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), train_data)

        return train_data, test_data

    def normalize(self, sq):
        d = sq.reshape(-1, self.var_num)
        d = self.scaler.transform(d)
        if self.auto_norm:
            d = normalize_to_neg_one_to_one(d)
        return d.reshape(-1, self.window, self.var_num)

    def unnormalize(self, sq):
        d = self.__unnormalize(sq.reshape(-1, self.var_num))
        return d.reshape(-1, self.window, self.var_num)
    
    def __normalize(self, rawdata):
        data = self.scaler.transform(rawdata)
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(data)
        return data

    def __unnormalize(self, data):
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        x = data
        return self.scaler.inverse_transform(x)
    
    @staticmethod
    def divide(data, ratio, seed=2023):
        size = data.shape[0]
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        regular_train_num = int(np.ceil(size * ratio))
        #id_rdm = np.random.permutation(size)
        id_rdm = np.arange(size)
        regular_train_id = id_rdm[:regular_train_num]
        irregular_train_id = id_rdm[regular_train_num:]

        regular_data = data[regular_train_id, :]
        irregular_data = data[irregular_train_id, :]

        # Restore RNG.
        np.random.set_state(st0)
        return regular_data, irregular_data

    @staticmethod
    def read_data(filepath, name=''):
        """Reads a single .csv
        """
        df = pd.read_csv(filepath, header=0)
        if name == 'etth':
            df.drop(df.columns[0], axis=1, inplace=True)
        data = df.values
        #scaler = MinMaxScaler()
        scaler = StandardScaler()
        scaler = scaler.fit(data)
        return data, scaler
    
    def mask_data(self, seed=2023):
        masks = np.ones_like(self.samples)
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        for idx in range(self.samples.shape[0]):
            x = self.samples[idx, :, :]  # (seq_length, feat_dim) array
            mask = noise_mask(x, self.missing_ratio, self.mean_mask_length, self.style,
                              self.distribution)  # (seq_length, feat_dim) boolean array
            masks[idx, :, :] = mask

        if self.save2npy:
            np.save(os.path.join(self.dir, f"{self.name}_masking_{self.window}.npy"), masks)

        # Restore RNG.
        np.random.set_state(st0)
        return masks.astype(bool)

    def __getitem__(self, ind):
        if self.period == 'test':
            x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
            m = self.masking[ind, :, :]  # (seq_length, feat_dim) boolean array
            return torch.from_numpy(x).float(), torch.from_numpy(m)
        x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
        return torch.from_numpy(x).float()

    def __len__(self):
        return self.sample_num

class CollapseRareLevels(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=1000):
        super().__init__()
        self.threshold = threshold
        self.level_maps = {}

    def fit(self, X, y=None):
        for col in X.select_dtypes(include=['object', 'category']).columns:
            counts = X[col].value_counts()
            if len(counts) > self.threshold:
                top_levels = counts.nlargest(self.threshold).index
                self.level_maps[col] = set(top_levels)
            else:
                self.level_maps[col] = None
        return self

    def transform(self, X):
        X_ = X.copy()
        for col, top_levels in self.level_maps.items():
            if top_levels is not None:
                X_[col] = X_[col].apply(lambda x: x if x in top_levels else 'RARE_LEVEL')
        return X_

class HistogramImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
        self.histograms_ = None 

    def fit(self, X, y=None):
        self.histograms_ = {}  # Initialize as empty dict
        for col in X.columns:
            non_na = X[col].dropna()
            self.histograms_[col] = non_na.values
        return self

    def transform(self, X):
        X_ = X.copy()
        rng = np.random.default_rng()
        for col in X.columns:
            mask = X_[col].isna()
            if mask.any() and col in self.histograms_:
                sampled = rng.choice(self.histograms_[col], size=mask.sum(), replace=True)
                X_.loc[mask, col] = sampled
        return X_
    
def make_openml_preprocessor(X_train_raw, include_onehot):
    # Identify categorical columns
    categorical_cols = X_train_raw.select_dtypes(['object', 'category']).columns.tolist()
    numerical_cols = [col for col in X_train_raw.columns if col not in categorical_cols]

    # Preprocessing pipelines
    return ColumnTransformer([
        ('cat', Pipeline([
            ('collapse', CollapseRareLevels(threshold=1000)),
            ('impute', SimpleImputer(strategy='constant', fill_value='MISSING')),
            *([('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))] if include_onehot else [])
        ]), categorical_cols),
        ('num', Pipeline([
            ('impute', HistogramImputer())
        ]), numerical_cols)
    ])
