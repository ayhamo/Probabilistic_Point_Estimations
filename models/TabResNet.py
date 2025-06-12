from configs.logger_config import global_logger as logger
from configs.config import DATASETS
from utils.data_loader import load_preprocessed_data
import utils.evaluation as evaluation

import torch
import torchmetrics
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from pytorch_widedeep import Trainer
from pytorch_widedeep.models import TabResnet, WideDeep
from pytorch_widedeep.preprocessing import TabPreprocessor

import numpy as np
import pandas as pd
import math

def initialize_train_tabresnet_regressor(model, X_train_tab_processed, y_train_tensor, training_params, device):
    """
    Initializes the Trainer and fits the TabResNet model.
    """
    
    logger.info(f"Initializing and training TabResNet Regressor...")

    optimizer = AdamW(model.parameters(), lr=training_params.get("lr", 0.001))

    # Define your training parameters
    n_epochs = training_params["n_epochs"]
    batch_size = training_params["batch_size"]

    # Create the OneCycleLR scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=training_params["max_lr"],
        epochs=n_epochs,
        steps_per_epoch= math.ceil(len(X_train_tab_processed) / batch_size),
    )

    trainer = Trainer(
        model=model,
        objective="regression",
        metrics= [torchmetrics.MeanSquaredError(squared=False).to(device)],
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_workers=0
    )

    trainer.fit(
        X_tab=X_train_tab_processed,
        target=y_train_tensor,
        n_epochs=n_epochs,
        batch_size=batch_size,
        val_split=0.2, # always have validation of 20%
        early_stop_patience=training_params["early_stop_patience"],
    )
    return trainer

def tabresnet_nll(y_test, y_pred):
    """
    Calculates the model-based Negative Log-Likelihood (NLL) assuming a Gaussian distribution
    for the target variable with variance estimated from the residuals.
    """
    
    residuals = y_test - y_pred
    variance = np.var(residuals)
    
    if variance <= 1e-6:
        logger.info(f"Estimated variance is close to zero. NLL might be unstable or Inf. Variance set to 1e-6.")
        variance = 1e-6 
    
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

def evaluate_tabresnet_model(trainer, X_test_processed, y_test):
    """
    Evaluates a TabResNet model, calculating regression metrics and NLL.
    """

    trainer.model.eval()
    
    y_pred = trainer.predict(X_tab=X_test_processed, batch_size=16)

    regression_metrics = evaluation.calculate_regression_metrics(y_test, y_pred)
    logger.info(f"Test Regression Metrics: {regression_metrics}")

    avg_nll = tabresnet_nll(y_test, y_pred)
    logger.info(f"TabResNet Model Test Mean NLL: {avg_nll:.4f}")
        
    return avg_nll, regression_metrics

def run_TabResNet_pipeline(
    source_dataset: str = "openml_ctr23",
    test_datasets=None,
    base_model_save_path_template : str = None # "trained_models/tabresnet_best_{dataset_key}.pth" # Not used in this template
    ):
    """
    Runs the TabResNet model training and evaluation pipeline.
    most of the code was taken from https://github.com/jrzaurin/pytorch-widedeep/issues/216
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
        if dataset_key == "361268":
            # currently, this dataset crashes the program, due to illegal llegal memory access in cuda
            device = torch.device("cpu")
        else:
            # just reassign it to cuda or cpu again
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dataset_name = dataset_info_dict.get('name', dataset_key)
        
        if source_dataset == "uci":
            if dataset_key == "protein-tertiary-structure":
                num_folds_to_run = 5
            else:
                num_folds_to_run = 20
        elif source_dataset == "openml_ctr23":
            num_folds_to_run = 10

        # taken from https://jrzaurin.github.io/infinitoml/2021/05/28/pytorch-widedeep_iv.html
        tabresnet_model_arch_params = {
            'blocks_dims': [200, 100, 100],  # or input_dim
            'blocks_dropout': 0.3,
            'mlp_hidden_dims': [100, 50],
            "mlp_dropout": 0.1,
            "mlp_batchnorm": False,
            "mlp_batchnorm_last": False,
            "mlp_linear_first": False,
        }

        tabresnet_training_params = {
            'n_epochs': 100,
            "lr": 0.005,
            "batch_size": 128,
            "max_lr": 0.01,
            'early_stop_patience': 10, # was 30, but i reduced it
        }

        logger.info(f"===== Starting TabResNet {num_folds_to_run}-Fold Evaluation for: {dataset_name} ({dataset_key}) =====")

        dataset_fold_metrics = {'nll': [], 'mae': [], 'mse': [], 'rmse': [], 'mape': []}

        for fold_idx in range(num_folds_to_run):
            logger.info(f"--- Processing Fold {fold_idx+1}/{num_folds_to_run} for dataset: {dataset_key} ---")
            
            # usually dataloader would load numpy, but in this model case we return a dataframe
            X_train_df, y_train_df, X_test_df, y_test = \
                load_preprocessed_data("TabResNet", source_dataset, dataset_key, fold_idx,
                        batch_size=0, # we return raw data, not pytorch loader here
                        openml_pre_prcoess=False) # Ensure this can return DataFrames

            categorical_cols = X_train_df.select_dtypes(['object', 'category']).columns.tolist()
            numerical_cols = [col for col in X_train_df.columns if col not in categorical_cols]

            embed_cols_for_preprocessor = [str(col) for col in categorical_cols] if categorical_cols else None

            tab_preprocessor = TabPreprocessor(
                embed_cols=embed_cols_for_preprocessor,
                continuous_cols=numerical_cols,
                normalize_continuous=True,
            )
            X_train_processed = tab_preprocessor.fit_transform(X_train_df)
            X_test_processed = tab_preprocessor.transform(X_test_df)

            X_train_processed = torch.tensor(X_train_processed, dtype=torch.float32).to(device)
            y_train_tensor = torch.tensor(y_train_df, dtype=torch.float32).reshape(-1, 1).to(device)

            deeptabular_model = TabResnet(
                column_idx=tab_preprocessor.column_idx,
                cat_embed_input= getattr(tab_preprocessor, "cat_embed_input", None),  
                continuous_cols=tab_preprocessor.continuous_cols,
                **tabresnet_model_arch_params
            )
            # Wrap the TabResNet component with WideDeep 
            # Output dim of WideDeep head should be 1 for regression.
            # The head is automatically added by WideDeep if pred_dim is not None for deeptabular.
            # Or TabResNet itself can have a final MLP. Here, WideDeep manages the final output layer.
            model_tabresnet = WideDeep(deeptabular=deeptabular_model, pred_dim=1)
            model_tabresnet = model_tabresnet.to(device)

            logger.info(f"Starting TabResNet training for {dataset_name}, Fold {fold_idx}...")
            trainer = initialize_train_tabresnet_regressor(
                model_tabresnet,
                X_train_processed,
                y_train_tensor,
                training_params=tabresnet_training_params,
                device=device # this is to fix 
            )
            logger.info("TabResNet training finished.")

            logger.info(f"Evaluating TabResNet on test set for {dataset_name}, Fold {fold_idx}...")

            avg_nll, reg_metrics = evaluate_tabresnet_model(
                trainer, X_test_processed, y_test)

            dataset_fold_metrics['nll'].append(avg_nll)
            dataset_fold_metrics['mae'].append(reg_metrics.get('MAE', np.nan))
            dataset_fold_metrics['mse'].append(reg_metrics.get('MSE', np.nan))
            dataset_fold_metrics['rmse'].append(reg_metrics.get('RMSE', np.nan))
            dataset_fold_metrics['mape'].append(reg_metrics.get('MAPE', np.nan))

            # Clean up to free memory
            del model_tabresnet, trainer, X_train_processed, X_test_processed, y_train_tensor
            if 'cuda' in device.type:
                torch.cuda.empty_cache()
       
        # AGGREGATED RESULTS for the current dataset
        logger.info(f"===== AGGREGATED TabResNet RESULTS for {dataset_name} ({dataset_key}) over {num_folds_to_run} Folds =====")
        
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

        broken_folds_indicator = f" *({broken_folds_count} broken/inf NLL folds)" if broken_folds_count > 0 else ""
        
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
    logger.info("===== ***** SUMMARY OF ALL TABRESNET DATASET EVALUATIONS ***** =====")
    for ds_key, results in overall_results_summary.items():
        logger.info(f"--- Dataset: {results['display_name']} ({ds_key}) ({results['num_folds']} Folds) ---")
        logger.info(f"  Average Test NLL: {results['NLL_mean']:.4f} ± {results['NLL_std']:.4f}")
        logger.info(f"  Average Test MSE: {results['MSE_mean']:.4f} ± {results['MSE_std']:.4f}")
        logger.info(f"  Average Test RMSE: {results['RMSE_mean']:.4f} ± {results['RMSE_std']:.4f}")
        logger.info(f"  Average Test MAE: {results['MAE_mean']:.4f} ± {results['MAE_std']:.4f}")
        logger.info(f"  Average Test MAPE: {results['MAPE_mean']:.2f}% ± {results['MAPE_std']:.2f}%\n")
    logger.info("===== ***** END OF TABRESNET OVERALL SUMMARY ***** =====")

    results_df = pd.DataFrame.from_dict(overall_results_summary, orient='index')
    return results_df