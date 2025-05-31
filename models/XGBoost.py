from configs.logger_config import global_logger as logger
from configs.config import DATASETS, RANDOM_STATE, device
from utils.data_loader import load_preprocessed_data
import utils.evaluation as evaluation

from xgboost import XGBRegressor
import numpy as np
import pandas as pd

def initialize_train_xgboost_regressor(X_train, y_train, **kwargs):
    logger.info(f"Initializing and training XGBoost Regressor with params: {kwargs}...")
    
    kwargs['random_state'] = RANDOM_STATE
        
    model = XGBRegressor(**kwargs)

    # XGBoost can handle Pandas DataFrames/Series or NumPy arrays
    model.fit(X_train, y_train)
    return model

def xgboost_nll(xgboost_regressor_model, X_test, y_test):
    """
    Calculates the model-based Negative Log-Likelihood (NLL) for a fitted XGBoost Regressor,
    assuming a Gaussian distribution for the target variable with variance estimated
    from the test set residuals.
    Returns a dictionary with 'Mean NLL' and 'Total NLL'.
    """
    results_nll = {'Mean NLL': 0, 'Total NLL': 0}

    y_test_values = y_test.values if isinstance(y_test, pd.Series) else np.asarray(y_test)
    
    y_pred = xgboost_regressor_model.predict(X_test)

    y_pred = np.asarray(y_pred)

    residuals = y_test_values - y_pred
    variance = np.var(residuals) 

    if variance <= 1e-9:
        logger.info("Estimated variance is close to zero in xgboost_nll. NLL might be unstable or Inf.")
        variance = 1e-9

    log_2pi = np.log(2 * np.pi)
    nll_per_sample = 0.5 * (log_2pi + np.log(variance) + (residuals**2) / variance)
    
    nll_per_sample_finite = nll_per_sample[np.isfinite(nll_per_sample)]


    results_nll['Mean NLL'] = np.mean(nll_per_sample_finite)
    results_nll['Total NLL'] = np.sum(nll_per_sample_finite)

        
    return results_nll

def evaluate_xgboost_model(model, X_test, y_test, y_pred):
    """
    Evaluates an XGBoost model, calculating regression metrics.
    NLL is currently returned as NaN.

    Args:
        model: The trained XGBoost model.
        X_test: Test features.
        y_test: Test target.
        y_pred: Predictions from the model.
        fold_idx: Current fold index (for logging).
        model_key: Identifier for the model (for logging).

    Returns:
        A tuple of (nll_metrics, regression_metrics) dictionaries.
    """
    regression_metrics = evaluation.calculate_regression_metrics(y_test, y_pred)
    logger.info(f"Test Regression Metrics: {regression_metrics}")

    nll_metrics = xgboost_nll(model, X_test, y_test)

    logger.info(f"xgboost_regressor Model Test Mean NLL: {nll_metrics['Mean NLL']:.4f}")
    logger.info(f"xgboost_regressor Model Test Total NLL: {nll_metrics['Total NLL']:.4f}\n")
        
    return nll_metrics, regression_metrics

def run_XGBoost_pipeline(
    source_dataset: str = "openml_ctr23",
    test_single_dataset: str = None,
    base_model_save_path_template : str = None # "trained_models/xgboost_best_{dataset_key}.pth"
    ):
    """
    Runs the XGBoost model training and evaluation pipeline.

    Args:
        source_dataset_type (str): The source of the datasets (e.g., "openml_ctr23").
        datasets_to_process : provide single dataset name configred in config.py to test model on
        base_model_save_path_template (str): A template string for loading pre-trained models
                                                      Example: "trained_models/xgboost_best_{dataset_key}.pth"


    Returns:
        pandas.DataFrame: A DataFrame summarizing the evaluation results across all processed datasets.
    """

    # For looping through all datasets in the source
    datasets_to_run = DATASETS.get(source_dataset, {})
    overall_results_summary = {}
    
    if test_single_dataset:
        datasets_to_run = DATASETS.get(source_dataset, {}).get(test_single_dataset, None)
        if datasets_to_run:
            datasets_to_run = {test_single_dataset: datasets_to_run}
        else:
            print("Could not find a default dataset for testing. Please check DATASETS structure.")
            datasets_to_run = {}

    overall_results_summary = {} # To store aggregated results for each dataset
    
    for dataset_key, dataset_info_dict in datasets_to_run.items():
        dataset_name = dataset_info_dict.get('name', dataset_key)
        
        if source_dataset == "uci":
            if dataset_key == "protein-tertiary-structure":
                num_folds_to_run = 5
            else:
                num_folds_to_run = 20 # 20
        elif source_dataset == "openml_ctr23":
            num_folds_to_run = 10 # 10

        # taken from OpenML-CTR23 paper
        xgboost_params_for_dataset = {
            'device' : 'cuda',
            'learning_rate': 0.1,            # Higher side from [1e-4, 1]
            'n_estimators': 3000,            # Higher side from [1, 5000]
            'max_depth': 15,                 # Higher side from [1, 20]
            'subsample': 0.7,                # Higher side from [0.1, 1]
            'colsample_bytree': 0.7,         # Higher side from [0.1, 1]
            'colsample_bylevel': 0.7,        # Higher side from [0.1, 1]
            'reg_alpha': 0.1,                # L1 reg, higher side from [0.001, 1000]
            'reg_lambda': 1.0,               # L2 reg, higher side from [0.001, 1000] (XGBoost default is 1)
        }

        logger.info(f"===== Starting XGBoost {num_folds_to_run}-Fold Evaluation for: {dataset_name} ({dataset_key}) =====")

        dataset_fold_metrics = {'nll': [], 'mae': [], 'mse': [], 'rmse': [], 'mape': []}

        for fold_idx in range(num_folds_to_run):
            logger.info(f"--- Processing Fold {fold_idx+1}/{num_folds_to_run} for dataset: {dataset_key} ---")
            
            X_train, y_train, X_test, y_test = \
                load_preprocessed_data("XGBoost", source_dataset, dataset_key, fold_idx,
                        batch_size=0, # batch_size is not used, so 0
                        openml_pre_prcoess=False)

            logger.info(f"Starting XGBoost training for {dataset_name}, Fold {fold_idx+1}...")
            model_xgb = initialize_train_xgboost_regressor(X_train, y_train, **xgboost_params_for_dataset)
            logger.info("XGBoost Regressor training finished.")

            logger.info(f"Evaluating XGBoost on test set for {dataset_name}, Fold {fold_idx+1}...")
            y_pred = model_xgb.predict(X_test)
            
            nll_metrics, reg_metrics = evaluate_xgboost_model(
                model_xgb, X_test, y_test, y_pred,
            )

            dataset_fold_metrics['nll'].append(nll_metrics.get('Mean NLL', np.nan))
            dataset_fold_metrics['mae'].append(reg_metrics.get('MAE', np.nan))
            dataset_fold_metrics['mse'].append(reg_metrics.get('MSE', np.nan))
            dataset_fold_metrics['rmse'].append(reg_metrics.get('RMSE', np.nan))
            dataset_fold_metrics['mape'].append(reg_metrics.get('MAPE', np.nan))

       
        # AGGREGATED RESULTS for the current dataset
        logger.info(f"===== AGGREGATED XGBoost RESULTS for {dataset_name} ({dataset_key}) over {num_folds_to_run} Folds =====")
        
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
            'display_name': dataset_name,
            'num_folds': num_folds_to_run,
            'NLL_mean': mean_nll, 'NLL_std': std_nll,
            'MSE_mean': mean_mse, 'MSE_std': std_mse,
            'RMSE_mean': mean_rmse, 'RMSE_std': std_rmse,
            'MAE_mean': mean_mae, 'MAE_std': std_mae,
            'MAPE_mean': mean_mape, 'MAPE_std': std_mape,
        }
        logger.info("===================================================================\n")
    
    # Final Summary for ALL Datasets
    logger.info("===== ***** SUMMARY OF ALL DATASET EVALUATIONS ***** =====")
    for ds_key, results in overall_results_summary.items():
        logger.info(f"--- Dataset: {results['display_name']} ({ds_key}) ({results['num_folds']} Folds) ---")
        logger.info(f"  Average Test NLL: {results['NLL_mean']:.4f} ± {results['NLL_std']:.4f}")
        logger.info(f"  Average Test MSE: {results['MSE_mean']:.4f} ± {results['MSE_std']:.4f}")
        logger.info(f"  Average Test RMSE: {results['RMSE_mean']:.4f} ± {results['RMSE_std']:.4f}")
        logger.info(f"  Average Test MAE: {results['MAE_mean']:.4f} ± {results['MAE_std']:.4f}")
        logger.info(f"  Average Test MAPE: {results['MAPE_mean']:.2f}% ± {results['MAPE_std']:.2f}%\n")
    logger.info("===== ***** END OF OVERALL SUMMARY ***** =====")

    results_df = pd.DataFrame.from_dict(overall_results_summary, orient='index')
    return results_df
    

def run_xgboost_optuna(
    source_dataset: str,
    datasets_to_optimize: list,
    n_trials_optuna: int = 100,
    hpo_fold_idx: int = 0,
    metric_to_optimize: str = "Mean NLL" # Can be "Mean NLL" or "RMSE"
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