from configs.logger_config import global_logger as logger
from configs.config import DATASETS, RANDOM_STATE, device
from models.TTVAEMain.ttvae.model import TTVAE
from utils.data_loader import load_preprocessed_data

import numpy as np
import pandas as pd
import os
import torch

def initialize_train_TTVAE_regressor(train_df, cat_columns_names, num_epochs, ckpt_path, **kwargs):
    logger.info(f"Initializing and training TTVAE Regressor...")

    model = TTVAE(verbose=True, epochs=num_epochs, device=device)
    
    model.fit(train_df, cat_columns_names, ckpt_path)

    return model

def evaluate_VAE_model(model, test_df, ckpt_path):
    """
    Evaluates an VAE model, calculating regression metrics.
    NLL is currently returned as NaN.

    Args:
        model: The trained XGBoost model.
        X_test: Test features.
        y_test: Test target.
        y_pred: Predictions from the model.

    Returns:
        A tuple of (nll_metrics, regression_metrics) dictionaries.
    """

    logger.info("--- Evaluating Model on Test Data using CRPS ---")

    # for now just use one from memory, no need to load it again!
    #best_model = torch.load(ckpt_path + '/model.pt', weights_only=False, map_location=device)
    best_model = model
    best_model.set_device(device)

    average_crps =best_model.estimate_crps(test_df, test_df.columns[-1])

    logger.info(f"Model Test Mean CRPS: {average_crps:.6f}")

    logger.info("--- Estimating Negative Log-Likelihood on Test Data ---")

    avg_nll = best_model.estimate_nll(test_df, n_samples=500)

    logger.info(f"Model Test Mean NLL: {avg_nll:.4f} ---")
        
    return average_crps, avg_nll

def run_TTVAE_pipeline(
    source_dataset: str = "openml_ctr23",
    test_datasets = None,
    epochs=60
    ):
    """
    Runs the VAE model training and evaluation pipeline.

    Args:
        source_dataset_type (str): The source of the datasets (e.g., "openml_ctr23").
        test_datasets (list) : provide a list of dataset name configred in config.py to test model on


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

        logger.info(f"===== Starting TTVAE {num_folds_to_run}-Fold Evaluation for: {dataset_name} ({dataset_key}) =====")

        dataset_fold_metrics = {'nll': [], 'crps': []}

        for fold_idx in range(num_folds_to_run):
            logger.info(f"--- Processing Fold {fold_idx+1}/{num_folds_to_run} for dataset: {dataset_key} ---")
            
            train_df, test_df = \
                load_preprocessed_data("TTVAE", source_dataset, dataset_key, fold_idx,
                        batch_size=0, # batch_size is not used, so 0
                        openml_pre_prcoess=True)
            
            # Assign column names as strings starting from "0", and MUST ADD header=None!
            train_df.columns = [str(i) for i in range(train_df.shape[1])]
            test_df.columns = [str(i) for i in range(test_df.shape[1])]

            logger.info(f"Starting TTVAE training for {dataset_name}, Fold {fold_idx}...")

            ckpt_path = f'trained_models/TTVAE/{dataset_name}/{fold_idx}'
            if not os.path.exists(ckpt_path):
                os.makedirs(ckpt_path)
                                                            # Cat Columns
                                    # always none as UCI has none and openML is pre-procssed in exact way!
            model = initialize_train_TTVAE_regressor(train_df, [], epochs, ckpt_path)

            logger.info("TTVAE Regressor training finished.")

            logger.info(f"Evaluating TTVAE on test set for {dataset_name}, Fold {fold_idx}...")
            
            crps, avg_nll = evaluate_VAE_model(model, test_df, ckpt_path)

            dataset_fold_metrics['nll'].append(avg_nll)
            dataset_fold_metrics['crps'].append(crps)

       
        # AGGREGATED RESULTS for the current dataset
        logger.info(f"===== AGGREGATED TTVAE RESULTS for {dataset_name} ({dataset_key}) over {num_folds_to_run} Folds =====")
        
        # Filter out broken NLL folds (inf values)
        valid_nll_values = [nll for nll in dataset_fold_metrics['nll'] if nll < 1e20]
        broken_folds_count = len(dataset_fold_metrics['nll']) - len(valid_nll_values)

        mean_nll = np.nanmean(valid_nll_values) if valid_nll_values else np.nan
        std_nll = np.nanstd(valid_nll_values) if valid_nll_values else np.nan
        mean_crps = np.nanmean(dataset_fold_metrics['crps'])
        std_crps = np.nanstd(dataset_fold_metrics['crps'])

        # Append '*' if there were broken folds
        broken_folds_indicator = f" *({broken_folds_count} broken folds)" if broken_folds_count > 0 else ""
        
        logger.info(f"  Average Test NLL: {mean_nll:.4f} ± {std_nll:.4f}{broken_folds_indicator}")
        logger.info(f"  Average Test CRPS: {mean_crps:.4f} ± {std_crps:.4f}")

        overall_results_summary[dataset_key] = {
            'display_name': dataset_name,
            'num_folds': num_folds_to_run,
            'NLL_mean': mean_nll, 'NLL_std': std_nll,
            'CRPS_mean': mean_crps, 'MSE_std': std_crps,
        }
        logger.info("===================================================================\n")
    
    # Final Summary for ALL Datasets
    logger.info("===== ***** SUMMARY OF ALL DATASET EVALUATIONS ***** =====")
    for ds_key, results in overall_results_summary.items():
        logger.info(f"--- Dataset: {results['display_name']} ({ds_key}) ({results['num_folds']} Folds) ---")
        logger.info(f"  Average Test NLL: {results['NLL_mean']:.4f} ± {results['NLL_std']:.4f}")
        logger.info(f"  Average Test CRPS: {results['CRPS_mean']:.4f} ± {results['MSE_std']:.4f}")
    logger.info("===== ***** END OF OVERALL SUMMARY ***** =====")

    results_df = pd.DataFrame.from_dict(overall_results_summary, orient='index')
    return results_df