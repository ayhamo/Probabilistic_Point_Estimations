from configs.logger_config import global_logger as logger
from configs.config import DATASETS, RANDOM_STATE, device
from utils.data_loader import load_preprocessed_data
import utils.evaluation as evaluation

import numpy as np
import pandas as pd


import torch
from tabpfn import TabPFNRegressor
# Removed due not being needed, but code is kept
#from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor

def initialize_train_tabpfn_regressor(X_train, y_train, **kwargs):
    """
    Initializes, fits, and returns a TabPFNRegressor model.
    """
    logger.info("Initializing and training TabPFNRegressor...")
    regressor = TabPFNRegressor(random_state=RANDOM_STATE,device=device, **kwargs)
    regressor.fit(X_train, y_train)

    return regressor

def initialize_train_autotabpfn_regressor(X_train, y_train, **kwargs):
    """
    Initializes, fits, and returns an AutoTabPFNRegressor model.
    it includes "Post-hoc ensembling combines multiple TabPFN models into an ensemble", which yelids
    better results
    """

    logger.info(f"Initializing and training AutoTabPFNRegressor (device={device})...")
    #auto_regressor = AutoTabPFNRegressor(device=device, **kwargs)
    #auto_regressor.fit(X_train, y_train)

    return None #auto_regressor

def tabpfn_nll(tabpfn_regressor_model, X_test, y_test):
    """
    Calculates the model-based Negative Log-Likelihood (NLL) for a fitted TabPFNRegressor.
    Returns a single value: avg_nll (Mean NLL).
    """
    y_test_values = y_test.values if isinstance(y_test, pd.Series) else np.asarray(y_test)
    
    full_output = tabpfn_regressor_model.predict(X_test, output_type="full")
    predicted_logits = full_output['logits']    # This is on CPU
    criterion_object = full_output['criterion'] # Its .borders are on CPU

    # Ensure y_test_tensor is on CPU, consistent with predicted_logits and criterion_object.borders
    y_test_tensor = torch.tensor(y_test_values, dtype=torch.float32).to(predicted_logits.device)

    nll_per_sample = criterion_object.forward(logits=predicted_logits, y=y_test_tensor)
    nll_per_sample_cpu = nll_per_sample.cpu().detach().numpy()

    # Calculate Mean NLL
    avg_nll = np.mean(nll_per_sample_cpu)
    
    INF_NLL_PLACEHOLDER = 1e20
    # Check for NaN, Inf in avg_nll and replace if necessary
    if np.isnan(avg_nll):
        avg_nll = np.nan
    elif np.isinf(avg_nll):
        avg_nll = INF_NLL_PLACEHOLDER if avg_nll > 0 else -INF_NLL_PLACEHOLDER

    return avg_nll


def evaluate_tabpfn_model(model, X_test, y_test, y_pred, fold_idx, model_key):
    """
    Calculates the model-based Negative Log-Likelihood (NLL) for a fitted TabPFNRegressor.
    Returns a dictionary with 'Mean NLL' and 'Total NLL'.
    """

    regression_metrics = evaluation.calculate_regression_metrics(y_test, y_pred)
    logger.info(f"Fold {fold_idx+1} Test Regression Metrics: {regression_metrics}")

    if model_key != "autotabpfn_regressor":
        avg_nll = tabpfn_nll(model, X_test, y_test)

        logger.info(f"Fold {fold_idx+1} {model_key} Model Test Mean NLL: {avg_nll:.4f}")
    else:
        logger.info(f"{model_key} NLL score cannot be computed\n")
        avg_nll = 0

    return avg_nll, regression_metrics

def run_TabPFN_pipeline(
    source_dataset :str = "openml_ctr23",
    test_datasets = None,
    models_train_types = ["tabpfn_regressor", "autotabpfn_regressor"],
    base_model_save_path_template : str = None # "trained_models/tabpfn_best_{dataset_key}.pth"
    ):
    """
    Runs the TabPFN model training and evaluation pipeline.

    Args:
        source_dataset_type (str): The source of the datasets (e.g., "openml_ctr23").
        test_datasets (list) : provide a list of dataset name configred in config.py to test model on
        models_train_types (list) : a list that contains model to train , options include tabpfn_regressor and autotabpfn_regressor
        base_model_save_path_template (str): A template string for loading pre-trained models
                                                      Example: "trained_models/tabpfn_best_{dataset_key}.pth"

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

    overall_results_summary = {} # To store aggregated results for each dataset and model 
    # For final aggregation across all datasets
    final_aggregation_data = {model_type: {'RMSE_means': [], 'MAE_means': [], 'NLL_means': []} for model_type in models_train_types}
    
    for dataset_key, dataset_info_dict in datasets_to_run.items():
        dataset_name = dataset_info_dict.get('name', dataset_key)
        
        if source_dataset == "uci":
            if dataset_key == "protein-tertiary-structure":
                num_folds_to_run = 5
            else:
                num_folds_to_run = 20
        elif source_dataset == "openml_ctr23":
            num_folds_to_run = 10

        dataset_fold_metrics = {
            model_type: {
                'nll': [], 'mae': [], 'mse': [], 'rmse': [], 'mape': []
            } for model_type in models_train_types
        }

        logger.info(f"===== Starting TabPFN {num_folds_to_run}-Fold Evaluation for: {dataset_name} ({dataset_key}) =====")

        for fold_idx in range(num_folds_to_run):

            logger.info(f"--- Processing Fold {fold_idx+1}/{num_folds_to_run} for dataset: {dataset_key} ---")

            if "tabpfn_regressor" in models_train_types:
                model_key = "tabpfn_regressor"

                X_train, y_train, X_test, y_test = \
                    load_preprocessed_data("TabPFN", source_dataset, dataset_key, fold_idx,
                            batch_size=0, # batch_size is not used, so 0
                            openml_pre_prcoess=True)
                
                logger.info(f"Starting training process for {dataset_name}, Fold {fold_idx}...")

                model_tabpfn = initialize_train_tabpfn_regressor(X_train, y_train)
                logger.info("TabPFN Training finished.")

                logger.info(f"Evaluating on test set for {dataset_name}, Fold {fold_idx}:")
                y_pred = model_tabpfn.predict(X_test, output_type="mean") # Get mean predictions for standard metrics

                avg_nll, reg_metrics = evaluate_tabpfn_model(model_tabpfn, X_test, y_test, y_pred, fold_idx, model_key = model_key)

                dataset_fold_metrics[model_key]['nll'].append(avg_nll)
                dataset_fold_metrics[model_key]['mae'].append(reg_metrics.get('MAE', np.nan))
                dataset_fold_metrics[model_key]['mse'].append(reg_metrics.get('MSE', np.nan))
                dataset_fold_metrics[model_key]['rmse'].append(reg_metrics.get('RMSE', np.nan))
                dataset_fold_metrics[model_key]['mape'].append(reg_metrics.get('MAPE', np.nan))

            if "autotabpfn_regressor" in models_train_types:
                model_key = "autotabpfn_regressor"

                X_train, y_train, X_test, y_test = \
                    load_preprocessed_data("TabPFN", source_dataset, dataset_key, fold_idx,
                            batch_size=0,
                            openml_pre_prcoess=True)
                    
                logger.info(f"Starting training process for {dataset_name}, Fold {fold_idx+1}...")

                model_autotabpfn = initialize_train_autotabpfn_regressor(X_train, y_train, device)
                logger.info("AutoTabPFN Training finished.")

                logger.info(f"Evaluating on test set for {dataset_name}, Fold {fold_idx+1}...")                
                y_pred = model_autotabpfn.predict(X_test)

                nll_metrics, reg_metrics = evaluate_tabpfn_model(model_autotabpfn, X_test, y_test, y_pred, fold_idx, model_key = model_key)

                dataset_fold_metrics[model_key]['nll'].append(nll_metrics.get('Mean NLL', np.nan))
                dataset_fold_metrics[model_key]['mae'].append(reg_metrics.get('MAE', np.nan))
                dataset_fold_metrics[model_key]['mse'].append(reg_metrics.get('MSE', np.nan))
                dataset_fold_metrics[model_key]['rmse'].append(reg_metrics.get('RMSE', np.nan))
                dataset_fold_metrics[model_key]['mape'].append(reg_metrics.get('MAPE', np.nan))


            
       # --- AGGREGATED RESULTS for the current dataset ---
        logger.info(f"===== AGGREGATED RESULTS for {dataset_name} ({dataset_key}) over {num_folds_to_run} Folds =====")
        overall_results_summary[dataset_key] = {
            'display_name': dataset_name,
            'num_folds': num_folds_to_run,
            'models': {}
        }

        for model_type, metrics_dict in dataset_fold_metrics.items():
            if not any(metrics_dict.values()): # Skip if no metrics were collected (e.g. model_type not run)
                continue

            # Filter out broken NLL folds
            valid_nll_values = [nll for nll in metrics_dict['nll'] if nll < 1e20]
            broken_folds_count = len(metrics_dict['nll']) - len(valid_nll_values)

            mean_nll = np.nanmean(valid_nll_values) if valid_nll_values else np.nan
            std_nll = np.nanstd(valid_nll_values) if valid_nll_values else np.nan
            mean_mse = np.nanmean(metrics_dict['mse']) if metrics_dict['mse'] else np.nan
            std_mse = np.nanstd(metrics_dict['mse']) if metrics_dict['mse'] else np.nan
            mean_rmse = np.nanmean(metrics_dict['rmse']) if metrics_dict['rmse'] else np.nan
            std_rmse = np.nanstd(metrics_dict['rmse']) if metrics_dict['rmse'] else np.nan
            mean_mae = np.nanmean(metrics_dict['mae']) if metrics_dict['mae'] else np.nan
            std_mae = np.nanstd(metrics_dict['mae']) if metrics_dict['mae'] else np.nan
            mean_mape = np.nanmean(metrics_dict['mape']) if metrics_dict['mape'] else np.nan
            std_mape = np.nanstd(metrics_dict['mape']) if metrics_dict['mape'] else np.nan

            # Append '*' if there were broken folds
            broken_folds_indicator = f" *({broken_folds_count} broken folds)" if broken_folds_count > 0 else ""

            logger.info(f"--- Model Type: {model_type} ---")
            logger.info(f"  Average Test NLL: {mean_nll:.4f} ± {std_nll:.4f}{broken_folds_indicator}")
            logger.info(f"  Average Test MSE: {mean_mse:.4f} ± {std_mse:.4f}")
            logger.info(f"  Average Test RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
            logger.info(f"  Average Test MAE: {mean_mae:.4f} ± {std_mae:.4f}")
            logger.info(f"  Average Test MAPE: {mean_mape:.2f}% ± {std_mape:.2f}%")

            overall_results_summary[dataset_key]['models'][model_type] = {
                'NLL_mean': mean_nll, 'NLL_std': std_nll,
                'MSE_mean': mean_mse, 'MSE_std': std_mse,
                'RMSE_mean': mean_rmse, 'RMSE_std': std_rmse,
                'MAE_mean': mean_mae, 'MAE_std': std_mae,
                'MAPE_mean': mean_mape, 'MAPE_std': std_mape,
                'broken_folds': broken_folds_count  # Keeping track of broken folds separately
            }
            # Store for final aggregation
            if model_type in final_aggregation_data:
                final_aggregation_data[model_type]['NLL_means'].append(mean_nll)
                final_aggregation_data[model_type]['RMSE_means'].append(mean_rmse)
                final_aggregation_data[model_type]['MAE_means'].append(mean_mae)
        logger.info("===================================================================\n")

    # --- Final Summary for ALL Datasets (Per Model Type) ---
    logger.info("===== ***** SUMMARY OF ALL DATASET EVALUATIONS ***** =====")
    for ds_key, results_data in overall_results_summary.items():
        logger.info(f"--- Dataset: {results_data['display_name']} ({ds_key}) ({results_data['num_folds']} Folds) ---")
        for model_type, metrics in results_data['models'].items():
            logger.info(f"  Model: {model_type}")
            logger.info(f"    Average Test NLL: {metrics['NLL_mean']:.4f} ± {metrics['NLL_std']:.4f}")
            logger.info(f"    Average Test MSE: {metrics['MSE_mean']:.4f} ± {metrics['MSE_std']:.4f}")
            logger.info(f"    Average Test RMSE: {metrics['RMSE_mean']:.4f} ± {metrics['RMSE_std']:.4f}")
            logger.info(f"    Average Test MAE: {metrics['MAE_mean']:.4f} ± {metrics['MAE_std']:.4f}")
            logger.info(f"    Average Test MAPE: {metrics['MAPE_mean']:.2f}% ± {metrics['MAPE_std']:.2f}%\n")

    logger.info("===== ***** SINGLE AGGREGATED SUMMARY ACROSS ALL DATASETS ***** =====")
    for model_type, agg_data in final_aggregation_data.items():
        if not agg_data['RMSE_means']: # Skip if no data for this model type
            continue
            
        grand_mean_nll = np.nanmean(agg_data['NLL_means'])
        grand_std_nll = np.nanstd(agg_data['NLL_means'])
        grand_mean_rmse = np.nanmean(agg_data['RMSE_means'])
        grand_std_rmse = np.nanstd(agg_data['RMSE_means'])
        grand_mean_mae = np.nanmean(agg_data['MAE_means'])
        grand_std_mae = np.nanstd(agg_data['MAE_means'])

        logger.info(f"--- Overall Aggregated Results for Model Type: {model_type} ---")
        logger.info(f"  Average of Mean NLLs across datasets: {grand_mean_nll:.4f} ± {grand_std_nll:.4f}")
        logger.info(f"  Average of Mean RMSEs across datasets: {grand_mean_rmse:.4f} ± {grand_std_rmse:.4f}")
        logger.info(f"  Average of Mean MAEs across datasets: {grand_mean_mae:.4f} ± {grand_std_mae:.4f}")
    logger.info("===== ***** END OF OVERALL SUMMARY ***** =====")
    
    # Create a DataFrame from the detailed summary
    flat_results = []
    for ds_key, data in overall_results_summary.items():
        for model_type, metrics in data.get('models', {}).items():
            row = {'dataset_key': ds_key, 'dataset_display_name': data['display_name'],
                   'num_folds': data['num_folds'], 'model_type': model_type, **metrics}
            flat_results.append(row)
    results_df = pd.DataFrame(flat_results)
    
    return results_df