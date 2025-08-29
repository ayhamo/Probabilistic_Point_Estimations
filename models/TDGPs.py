from configs.logger_config import global_logger as logger
from configs.config import DATASETS, RANDOM_STATE, device
from utils.data_loader import load_preprocessed_data
import utils.evaluation as evaluation

# TDGP-specific imports
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Libraries from the TDGP paper's codebase
import gpflow
from models.thindeepgps import tdgplib

from models.thindeepgps.tdgplib import helper
from models.thindeepgps.tdgplib import models
from models.thindeepgps.tdgplib import training

import uq360.metrics

# Disable verbose TensorFlow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

def initialize_train_tdgp(X_train, y_train, **kwargs):
    """
    Initializes and trains a Thin and Deep Gaussian Process (TDGP) model.
    """
    logger.info("Initializing and training TDGP model...")
    
    # model hyperparameters
    num_inducing_v = kwargs.get('num_inducing_v', 25)
    num_inducing_u = kwargs.get('num_inducing_u', 50)
    num_latent_q = kwargs.get('num_latent_q', X_train.shape[1]) # Defaults to input dimension D
    
    n, D = X_train.shape
    Q = num_latent_q
    
    # shapes for inducing variables
    Z_v = (num_inducing_v, D)
    Z_u = (num_inducing_u, Q)

    # TDGP model from the paper (referred to as vmgp in their example)
    model = models.TDGP(
        (X_train, y_train), Z_v, Z_u,[gpflow.kernels.RBF(lengthscales=np.ones(D)) for _ in range(Q)], True, False
    )
    
    # dummy monitor to suppress verbose training output
    dummy_monitor = gpflow.monitor.Monitor(gpflow.monitor.MonitorTaskGroup(
        gpflow.monitor.ExecuteCallback(lambda *args, **kwargs: None),
        period=10000
    ))

    # Train the model
    training.train_model(model, dummy_monitor, train_data=(X_train, y_train))
    
    logger.info("TDGP model training finished.")
    return model

def evaluate_tdgp_model(model, X_test, y_test, y_scaler):
    """
    Evaluates a trained TDGP model, calculating NLL and standard regression metrics.
    
    Args:
        model: The trained TDGP model.
        X_test: Scaled test features.
        y_test: Scaled test target.
        y_scaler: The fitted scaler for the target variable to inverse-transform results.
        
    Returns:
        A tuple of (avg_nll, regression_metrics, y_pred_unscaled).
    """
    logger.info("Evaluating TDGP model on the test set...")
    
    # Make Predictions (in scaled space)
    likelihood_variance = tdgplib.helper.get_likelihood_variance(model)
    f_pred_mean, f_pred_var = model.predict_f(X_test)
    
    # Handle multi-sample predictions if applicable
    if len(f_pred_mean.shape) == 3:
        f_pred_mean = tf.reduce_mean(f_pred_mean, axis=0)
        f_pred_var = tf.reduce_mean(f_pred_var, axis=0)
        
    y_pred_mean_scaled = f_pred_mean
    y_pred_var_scaled = f_pred_var + likelihood_variance
    
    # --- Calculate NLL (in scaled space) ---
    try:
        nll_metrics = uq360.metrics.regression_metrics.compute_regression_metrics(
            y_test,
            y_mean=y_pred_mean_scaled.numpy(),
            y_lower=(y_pred_mean_scaled - 2 * np.sqrt(y_pred_var_scaled)).numpy(),
            y_upper=(y_pred_mean_scaled + 2 * np.sqrt(y_pred_var_scaled)).numpy(),
            option="nll"
        )
        avg_nll = nll_metrics.get('nll', np.nan)
    except ValueError as e:
        logger.error(f'Error computing NLL metric: {e}')
        avg_nll = np.nan
        
    logger.info(f"TDGP Model Test Mean NLL (scaled): {avg_nll:.4f}")

    # --- Calculate Regression Metrics (in original data space) ---
    # Inverse transform predictions and ground truth to original scale
    y_pred_unscaled = y_scaler.inverse_transform(y_pred_mean_scaled.numpy())
    y_test_unscaled = y_scaler.inverse_transform(y_test)
    
    regression_metrics = evaluation.calculate_regression_metrics(y_test_unscaled, y_pred_unscaled)
    logger.info(f"Test Regression Metrics (unscaled): {regression_metrics}")
    
    return avg_nll, regression_metrics, y_pred_unscaled

def run_TDGP_pipeline(
    source_dataset: str = "openml_ctr23",
    test_datasets=None,
    base_model_save_path_template: str = None
    ):
    """
    Runs the TDGP model training and evaluation pipeline.
    """
    datasets_to_run = {}
    
    if test_datasets:
        for dataset_key in test_datasets:
            dataset_value = DATASETS.get(source_dataset, {}).get(dataset_key, None)
            if dataset_value:
                datasets_to_run[dataset_key] = dataset_value
            else:
                logger.warning(f"Could not find dataset '{dataset_key}' in config.")
    else:
        datasets_to_run = DATASETS.get(source_dataset, {})

    overall_results_summary = {}
    
    for dataset_key, dataset_info_dict in datasets_to_run.items():
        dataset_name = dataset_info_dict.get('name', dataset_key)

        if source_dataset == "uci":
            if dataset_key == "protein-tertiary-structure":
                num_folds_to_run = 1#5
            else:
                num_folds_to_run = 1#20
        elif source_dataset == "openml_ctr23":
            num_folds_to_run = 1#10

        # TDGP model parameters
        tdgp_params = {
            'num_inducing_v': 25,  # Inducing points for the "thin" layer
            'num_inducing_u': 50,  # Inducing points for the output GP layer
            'num_latent_q': None   # Latent dimensions, will default to input dim
        }

        logger.info(f"===== Starting TDGP {num_folds_to_run}-Fold Evaluation for: {dataset_name} ({dataset_key}) =====")

        dataset_fold_metrics = {'nll': [], 'mae': [], 'mse': [], 'rmse': [], 'mape': []}

        for fold_idx in range(num_folds_to_run):
            logger.info(f"--- Processing Fold {fold_idx+1}/{num_folds_to_run} for dataset: {dataset_key} ---")
            
            helper.set_all_random_seeds(RANDOM_STATE)
            
            # Load data
            X_train, y_train, X_test, y_test = \
                load_preprocessed_data("TDGP", source_dataset, dataset_key, fold_idx,
                        batch_size=0,
                        openml_pre_prcoess=True)
            
            x_scaler = StandardScaler()
            y_scaler = StandardScaler()
            
            X_train_s = x_scaler.fit_transform(X_train)
            X_test_s = x_scaler.transform(X_test)
            
            y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1))
            y_test_s = y_scaler.transform(y_test.reshape(-1, 1))
            
            # Update params with the correct input dimension for this dataset
            if tdgp_params['num_latent_q'] is None:
                tdgp_params['num_latent_q'] = X_train_s.shape[1]

            model_tdgp = initialize_train_tdgp(X_train_s, y_train_s, **tdgp_params)
            
            avg_nll, reg_metrics, _ = evaluate_tdgp_model(
                model_tdgp, X_test_s, y_test_s, y_scaler
            )

            dataset_fold_metrics['nll'].append(avg_nll)
            dataset_fold_metrics['mae'].append(reg_metrics.get('MAE', np.nan))
            dataset_fold_metrics['mse'].append(reg_metrics.get('MSE', np.nan))
            dataset_fold_metrics['rmse'].append(reg_metrics.get('RMSE', np.nan))
            dataset_fold_metrics['mape'].append(reg_metrics.get('MAPE', np.nan))

        # AGGREGATED RESULTS for the current dataset
        logger.info(f"===== AGGREGATED TDGP RESULTS for {dataset_name} ({dataset_key}) over {num_folds_to_run} Folds =====")
        
        mean_nll = np.nanmean(dataset_fold_metrics['nll'])
        std_nll = np.nanstd(dataset_fold_metrics['nll'])
        mean_mse = np.nanmean(dataset_fold_metrics['mse'])
        std_mse = np.nanstd(dataset_fold_metrics['mse'])
        mean_rmse = np.nanmean(dataset_fold_metrics['rmse'])
        std_rmse = np.nanstd(dataset_fold_metrics['rmse'])
        mean_mae = np.nanmean(dataset_fold_metrics['mae'])
        std_mae = np.nanstd(dataset_fold_metrics['mae'])
        mean_mape = np.nanmean(dataset_fold_metrics['mape'])
        std_mape = np.nanstd(dataset_fold_metrics['mape'])

        logger.info(f"  Average Test NLL: {mean_nll:.4f} ± {std_nll:.4f}")
        logger.info(f"  Average Test MSE: {mean_mse:.4f} ± {std_mse:.4f}")
        logger.info(f"  Average Test RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
        logger.info(f"  Average Test MAE: {mean_mae:.4f} ± {std_mae:.4f}")
        logger.info(f"  Average Test MAPE: {mean_mape:.2f}% ± {std_mape:.2f}%")

        overall_results_summary[dataset_key] = {
            'display_name': dataset_name, 'num_folds': num_folds_to_run,
            'NLL_mean': mean_nll, 'NLL_std': std_nll, 'MSE_mean': mean_mse, 'MSE_std': std_mse,
            'RMSE_mean': mean_rmse, 'RMSE_std': std_rmse, 'MAE_mean': mean_mae, 'MAE_std': std_mae,
            'MAPE_mean': mean_mape, 'MAPE_std': std_mape,
        }
        logger.info("===================================================================\n")
    
    # Final Summary for ALL Datasets
    results_df = pd.DataFrame.from_dict(overall_results_summary, orient='index')
    logger.info("===== ***** FINAL SUMMARY OF TDGP EVALUATIONS ***** =====")
    print(results_df)
    logger.info("===== ***** END OF OVERALL SUMMARY ***** =====")

    return results_df

def run_xgboost_optuna(
    source_dataset: str,
    datasets_to_optimize: list,
    n_trials_optuna: int = 100,
    hpo_fold_idx: int = 0,
    metric_to_optimize: str = "RMSE" # Can be "Mean NLL" or "RMSE"
):
    """
    Runs Optuna hyperparameter optimization for XGBoost on specified datasets.

    Args:
        source_dataset (str): The source of the datasets (e.g., "openml_ctr23").
        datasets_to_optimize (list): A list of dataset keys (strings) to perform HPO on.
        n_trials_optuna (int): Number of Optuna trials to run for each dataset.
        hpo_fold_idx (int): The index of the fold to use for hyperparameter optimization.
                            Training will be on this fold's train set, validation on its test set.
        metric_to_optimize (str): The metric to optimize ("Mean NLL" or "RMSE").

    Returns:
        dict: A dictionary where keys are dataset_keys and values are dictionaries
              containing 'best_params' and 'best_value' from Optuna.
    """
    import optuna

    all_best_hyperparams = {}
    
    # Determine device once
    xgb_device = 'cuda' if device.type == 'cuda' else 'cpu'

    for dataset_key in datasets_to_optimize:
        dataset_info = DATASETS.get(source_dataset, {}).get(dataset_key, None)
        if not dataset_info:
            logger.warning(f"Dataset key '{dataset_key}' not found in source '{source_dataset}'. Skipping.")
            continue
        
        dataset_name = dataset_info.get('name', dataset_key)
        logger.info(f"===== Starting Optuna HPO for XGBoost on: {dataset_name} fold {hpo_fold_idx} ({dataset_key}) =====")
        logger.info(f"Optimizing for: {metric_to_optimize} over {n_trials_optuna} trials.")

        # Load data for HPO (using the specified fold)
        X_train, y_train, X_test, y_test = \
            load_preprocessed_data("XGBoost", source_dataset, dataset_key, hpo_fold_idx,
                                   batch_size=0, openml_pre_prcoess=True) # Corrected typo

        def objective(trial):
            # Define the search space for hyperparameters
            # These ranges are examples; adjust them based on your expectations/prior knowledge
            params = {
                'device': xgb_device,
                'random_state': RANDOM_STATE, # Consistent random state for XGBoost itself
                'verbosity': 0, # Suppress XGBoost's own messages during HPO                
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 500, 4000, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 15), # Original had 15, reduced upper for faster HPO
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True), # L1
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True), # L2
            }
            
            # Train model
            model_xgb = initialize_train_xgboost_regressor(X_train, y_train, **params)

            # Predict and evaluate
            y_pred = model_xgb.predict(X_test)
            reg_metrics = evaluate_xgboost_model(model_xgb, X_test, y_test, y_pred)

            merged_metrics = {**reg_metrics[0], **reg_metrics[1]}
            return merged_metrics[metric_to_optimize] if metric_to_optimize in merged_metrics else float('inf') 
        
        # Create an Optuna study
        study = optuna.create_study(direction='minimize') 

        optuna_original_verbosity = optuna.logging.get_verbosity()
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study.optimize(objective, n_trials=n_trials_optuna)

        optuna.logging.set_verbosity(optuna_original_verbosity)

        logger.info(f"\nOptuna HPO finished for {dataset_name} ({dataset_key}).")
        logger.info(f"  Best trial number: {study.best_trial.number}")
        logger.info(f"  Best {metric_to_optimize}: {study.best_value:.4f}")
        logger.info(f"  Best hyperparameters: {study.best_params}")
        
        # Store results, ensuring 'device' is included as it's fixed, not tuned
        best_params_with_device = study.best_params.copy()

        all_best_hyperparams[dataset_key] = {
            'best_params': best_params_with_device,
            'best_value': study.best_value
        }
        logger.info("===================================================================\n")

    logger.info("===== ***** SUMMARY OF OPTUNA HPO RESULTS ***** =====")
    for ds_key, result in all_best_hyperparams.items():
        dataset_name = DATASETS.get(source_dataset, {}).get(ds_key, {}).get('name', ds_key)
        logger.info(f"--- Dataset: {dataset_name} ({ds_key}) ---")
        logger.info(f"  Best {metric_to_optimize}: {result['best_value']:.4f}")
        logger.info(f"  Best Hyperparameters: {result['best_params']}\n")
    logger.info("===== ***** END OF OPTUNA HPO SUMMARY ***** =====")
    
    return all_best_hyperparams