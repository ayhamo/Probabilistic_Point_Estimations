from configs.logger_config import global_logger as logger
from configs.config import DATASETS
from utils.data_loader import load_preprocessed_data
import utils.evaluation as evaluation

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd

def initialize_train_catboost_regressor(X_train, y_train, categorical_features_indices, **kwargs):
    logger.info(f"Initializing and training CatBoost Regressor...")
    
    model = CatBoostRegressor(**kwargs)

    model.fit(X_train, y_train, cat_features=categorical_features_indices, verbose=kwargs.get('verbose', 0))

    return model

def catboost_nll(catboost_regressor_model, X_test, y_test):
    """
    Calculates the model-based Negative Log-Likelihood (NLL) for a fitted CatBoost Regressor,
    assuming a Gaussian distribution for the target variable with variance estimated
    from the test set residuals.
    Returns a dictionary with 'Mean NLL' and 'Total NLL'.
    """
    y_test_values = y_test.values if isinstance(y_test, pd.Series) else np.asarray(y_test)
    y_pred = catboost_regressor_model.predict(X_test)
    y_pred = np.asarray(y_pred)
    
    residuals = y_test_values - y_pred
    variance = np.var(residuals)
    
    if variance <= 1e-6: # https://docs.pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html
        logger.info("Estimated variance is close to zero in CatBoost_nll. NLL might be unstable or Inf.")
        variance = max(variance, 1e-6)
    
    log_2pi = np.log(2 * np.pi)
    nll_per_sample = 0.5 * (log_2pi + np.log(variance) + (residuals**2) / variance)
    nll_per_sample_finite = nll_per_sample[np.isfinite(nll_per_sample)]
    
    avg_nll = np.mean(nll_per_sample_finite)
    
    INF_NLL_PLACEHOLDER = 1e20
    # Check for NaN, Inf in avg_nll and replace if necessary
    if np.isnan(avg_nll):
        avg_nll = np.nan
    elif np.isinf(avg_nll):
        avg_nll = INF_NLL_PLACEHOLDER if avg_nll > 0 else -INF_NLL_PLACEHOLDER
    
    return avg_nll

def evaluate_catboost_model(model, X_test, y_test, y_pred):
    """
    Evaluates an CatBoost model, calculating regression metrics.
    NLL is currently returned as NaN.

    Args:
        model: The trained CatBoost model.
        X_test: Test features.
        y_test: Test target.
        y_pred: Predictions from the model.

    Returns:
        A tuple of (nll_metrics, regression_metrics) dictionaries.
    """


    regression_metrics = evaluation.calculate_regression_metrics(y_test, y_pred)
    logger.info(f"Test Regression Metrics: {regression_metrics}")

    avg_nll = catboost_nll(model, X_test, y_test)

    logger.info(f"CatBoost_regressor Model Test Mean NLL: {avg_nll:.4f}")
        
    return avg_nll, regression_metrics

def run_CatBoost_pipeline(
    source_dataset: str = "openml_ctr23",
    test_datasets = None,
    base_model_save_path_template : str = None # "trained_models/CatBoost_best_{dataset_key}.pth"
    ):
    """
    Runs the CatBoost model training and evaluation pipeline.

    Args:
        source_dataset_type (str): The source of the datasets (e.g., "openml_ctr23").
        test_datasets (list) : provide a list of dataset name configred in config.py to test model on
        base_model_save_path_template (str): A template string for loading pre-trained models
                                                      Example: "trained_models/catboost_best_{dataset_key}.pth"


    Returns:
        pandas.DataFrame: A DataFrame summarizing the evaluation results across all processed datasets.
    """

    datasets_to_run = {}
    
    # override all datasets if a test list is given
    if test_datasets:
        for dataset_key in test_datasets:
            dataset_value = DATASETS.get(source_dataset, {}).get(dataset_key, None)
            if dataset_value:
                datasets_to_run[dataset_key] = dataset_value
            else:
                print("Could not find a default dataset for testing. Please check DATASETS structure.")
                datasets_to_run = {}
    else:
        # For looping through all datasets in the source
        datasets_to_run = DATASETS.get(source_dataset, {})

    overall_results_summary = {} # To store aggregated results for each dataset
    
    for dataset_key, dataset_info_dict in datasets_to_run.items():
        dataset_name = dataset_info_dict.get('name', dataset_key)
        
        if source_dataset == "uci":
            if dataset_key == "protein-tertiary-structure":
                num_folds_to_run = 5
            else:
                num_folds_to_run = 20
        elif source_dataset == "openml_ctr23":
            num_folds_to_run = 10

        # the ranges are taken from catboost paper
        # general paramters are taken from https://github.com/catboost/benchmarks/blob/master/quality_benchmarks/comparison_description.pdf
        # as they say, out-of-the-box performance, so below are generally used, not optimized per dataset
        catboost_params_for_dataset = {
            "task_type" : "GPU",
            "loss_function" : 'RMSE',
            'learning_rate': 0.03,     # Log-uniform [e^07,1]
            'depth': 6,
            'fold_len_multiplier': 2,  # Discrete uniform [1,20]
            'l2_leaf_reg': 3,          # Log-uniform [1,10]
            'random_strength': 1,      # Discrete uniform [0,25]
            'one_hot_max_size': 0,     # Discrete uniform [0,25]
            'bagging_temperature': 1,  # Uniform [0,1]
        }

        logger.info(f"===== Starting CatBoost {num_folds_to_run}-Fold Evaluation for: {dataset_name} ({dataset_key}) =====")

        dataset_fold_metrics = {'nll': [], 'mae': [], 'mse': [], 'rmse': [], 'mape': []}

        for fold_idx in range(num_folds_to_run):
            logger.info(f"--- Processing Fold {fold_idx+1}/{num_folds_to_run} for dataset: {dataset_key} ---")
            
            X_train, y_train, X_test, y_test = \
                load_preprocessed_data("CatBoost", source_dataset, dataset_key, fold_idx,
                        batch_size=0, # batch_size is not used, so 0
                        openml_pre_prcoess=False) # CatBoost specfically ask not to one hot encode
            
            # have to convert both X train and test to pd frame, and then convert all non real numbers to str
            # otherwise catboost won't work and will crash, and it will give good result despite such pre-processing
            # since only openML datasets need this, i put it underneath that condition
            if source_dataset == "openml_ctr23":
                X_train = pd.DataFrame(X_train)
                categorical_features_indices = np.where(X_train.dtypes != float)[0]
                X_train[categorical_features_indices] = X_train[categorical_features_indices].astype(str)    

                X_test = pd.DataFrame(X_test)
                categorical_features_indices = np.where(X_test.dtypes != float)[0]
                X_test[categorical_features_indices] = X_test[categorical_features_indices].astype(str) 
            else:
                 # this is catboost default
                 categorical_features_indices = None

            logger.info(f"Starting CatBoost training for {dataset_name}, Fold {fold_idx}...")
            model_cb = initialize_train_catboost_regressor(X_train, y_train, categorical_features_indices, **catboost_params_for_dataset)
            logger.info("CatBoost Regressor training finished.")

            logger.info(f"Evaluating CatBoost on test set for {dataset_name}, Fold {fold_idx}...")
            
            y_pred = model_cb.predict(X_test)

            avg_nll, reg_metrics = evaluate_catboost_model(
                model_cb, X_test, y_test, y_pred,
            )

            dataset_fold_metrics['nll'].append(avg_nll)
            dataset_fold_metrics['mae'].append(reg_metrics.get('MAE', np.nan))
            dataset_fold_metrics['mse'].append(reg_metrics.get('MSE', np.nan))
            dataset_fold_metrics['rmse'].append(reg_metrics.get('RMSE', np.nan))
            dataset_fold_metrics['mape'].append(reg_metrics.get('MAPE', np.nan))

       
        # AGGREGATED RESULTS for the current dataset
        logger.info(f"===== AGGREGATED CatBoost RESULTS for {dataset_name} ({dataset_key}) over {num_folds_to_run} Folds =====")
        
        # Filter out broken NLL folds (inf values)
        valid_nll_values = [nll for nll in dataset_fold_metrics['nll'] if nll < 1e20]
        broken_folds_count = len(dataset_fold_metrics['nll']) - len(valid_nll_values)

        mean_nll = np.nanmean(valid_nll_values) if valid_nll_values else np.nan
        std_nll = np.nanstd(valid_nll_values) if valid_nll_values else np.nan
        mean_mse = np.nanmean(dataset_fold_metrics['mse'])
        std_mse = np.nanstd(dataset_fold_metrics['mse'])
        mean_rmse = np.nanmean(dataset_fold_metrics['rmse'])
        std_rmse = np.nanstd(dataset_fold_metrics['rmse'])
        mean_mae = np.nanmean(dataset_fold_metrics['mae'])
        std_mae = np.nanstd(dataset_fold_metrics['mae'])
        mean_mape = np.nanmean(dataset_fold_metrics['mape'])
        std_mape = np.nanstd(dataset_fold_metrics['mape'])
        
        # Append '*' if there were broken folds
        broken_folds_indicator = f" *({broken_folds_count} broken folds)" if broken_folds_count > 0 else ""
        
        logger.info(f"  Average Test NLL: {mean_nll:.4f} ± {std_nll:.4f}{broken_folds_indicator}")
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
