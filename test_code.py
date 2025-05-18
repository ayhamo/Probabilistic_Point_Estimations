from configs.logger_config import global_logger as logger
import os
import numpy as np

import torch
from configs.config import DATASETS

from models.point_TabResFlow import TabResFlow
from utils.data_loader import load_preprocessed_data
import utils.evaluation as evaluation

'''
TODO:


1. put all ctr23 datasets, and put a table FOR ALL DATASETS train size, test size, number of input features, number of output feautres, number of catorgical features, number missing feautres, percentage missing

2. look at kiran paper new and update
3. talk to kiran and see his implemtnation and compare with folding
4. implement tabpfn, xgboost, catboost

'''

# --- BEGIN: Dataset-specific configurations ---
ALL_DATASET_MODEL_CONFIGS = {
    "yacht": {
        "MODEL_HYPERPARAMS": {
            'embedding_dim_per_feature': 64, 'numerical_encoder_intermediate_dim': 100,
            'resnet_main_processing_dim': 128, 'resnet_depth': 3, 'resnet_block_hidden_factor': 1.0,
            'resnet_activation_dropout': 0.1, 'resnet_residual_dropout': 0.1,
            'flow_transforms': 3, 'flow_mlp_layers_in_transform': 2, 'flow_bins': 8,
            'categorical_cardinalities': None,
        },
        "TRAIN_HYPERPARAMS": {
            'lr': 0.003, 'weight_decay': 1e-4, 'num_epochs': 400,
            'patience_early_stopping': 400, 'batch_size': 1024
        }
    },
    "concrete": {
        "MODEL_HYPERPARAMS": {
            'embedding_dim_per_feature': 64, 'numerical_encoder_intermediate_dim': 100,
            'resnet_main_processing_dim': 256, 'resnet_depth': 6, 'resnet_block_hidden_factor': 1.0,
            'resnet_activation_dropout': 0.2, 'resnet_residual_dropout': 0.2,
            'flow_transforms': 5, 'flow_mlp_layers_in_transform': 2, 'flow_bins': 8,
            'categorical_cardinalities': None,
        },
        "TRAIN_HYPERPARAMS": {
            'lr': 0.003, 'weight_decay': 1e-4, 'num_epochs': 400,
            'patience_early_stopping': 400, 'batch_size': 2048
        }
    },
    "energy": {
        "MODEL_HYPERPARAMS": {
            'embedding_dim_per_feature': 64, 'numerical_encoder_intermediate_dim': 100,
            'resnet_main_processing_dim': 256, 'resnet_depth': 6, 'resnet_block_hidden_factor': 1.0,
            'resnet_activation_dropout': 0.2, 'resnet_residual_dropout': 0.2,
            'flow_transforms': 5, 'flow_mlp_layers_in_transform': 2, 'flow_bins': 8,
            'categorical_cardinalities': None,
        },
        "TRAIN_HYPERPARAMS": {
            'lr': 0.003, 'weight_decay': 1e-4, 'num_epochs': 400,
            'patience_early_stopping': 400, 'batch_size': 2024 # As per your original script
        }
    },
    "kin8nm": {
        "MODEL_HYPERPARAMS": {
            'embedding_dim_per_feature': 128, 'numerical_encoder_intermediate_dim': 100,
            'resnet_main_processing_dim': 256, 'resnet_depth': 6, 'resnet_block_hidden_factor': 1.0,
            'resnet_activation_dropout': 0.2, 'resnet_residual_dropout': 0.2,
            'flow_transforms': 5, 'flow_mlp_layers_in_transform': 2, 'flow_bins': 8,
            'categorical_cardinalities': None,
        },
        "TRAIN_HYPERPARAMS": {
            'lr': 0.003, 'weight_decay': 1e-4, 'num_epochs': 60,
            'patience_early_stopping': 50, 'batch_size': 2048
        }
    },
    "naval_propulsion_plant": {
        "MODEL_HYPERPARAMS": {
            'embedding_dim_per_feature': 128, 'numerical_encoder_intermediate_dim': 100,
            'resnet_main_processing_dim': 256, 'resnet_depth': 6, 'resnet_block_hidden_factor': 1.0,
            'resnet_activation_dropout': 0.2, 'resnet_residual_dropout': 0.2,
            'flow_transforms': 5, 'flow_mlp_layers_in_transform': 2, 'flow_bins': 8,
            'categorical_cardinalities': None,
        },
        "TRAIN_HYPERPARAMS": {
            'lr': 0.003, 'weight_decay': 1e-4, 'num_epochs': 300,
            'patience_early_stopping': 100, 'batch_size': 2048
        }
    },
    "power_plant": {
        "MODEL_HYPERPARAMS": {
            'embedding_dim_per_feature': 64, 'numerical_encoder_intermediate_dim': 100,
            'resnet_main_processing_dim': 256, 'resnet_depth': 4, 'resnet_block_hidden_factor': 1.0,
            'resnet_activation_dropout': 0.1, 'resnet_residual_dropout': 0.1,
            'flow_transforms': 4, 'flow_mlp_layers_in_transform': 2, 'flow_bins': 8,
            'categorical_cardinalities': None,
        },
        "TRAIN_HYPERPARAMS": {
            'lr': 0.003, 'weight_decay': 1e-4, 'num_epochs': 200,
            'patience_early_stopping': 100, 'batch_size': 2048
        }
    },
    "protein_tertiary_structure": {
        "MODEL_HYPERPARAMS": {
            'embedding_dim_per_feature': 128, 'numerical_encoder_intermediate_dim': 100,
            'resnet_main_processing_dim': 512, 'resnet_depth': 8, 'resnet_block_hidden_factor': 1.0,
            'resnet_activation_dropout': 0.3, 'resnet_residual_dropout': 0.3,
            'flow_transforms': 5, 'flow_mlp_layers_in_transform': 2, 'flow_bins': 10,
            'categorical_cardinalities': None,
        },
        "TRAIN_HYPERPARAMS": {
            'lr': 0.003, 'weight_decay': 1e-4, 'num_epochs': 100,
            'patience_early_stopping': 50, 'batch_size': 2048
        }
    },
    "wine_quality_red": {
        "MODEL_HYPERPARAMS": {
            'embedding_dim_per_feature': 64, 'numerical_encoder_intermediate_dim': 100,
            'resnet_main_processing_dim': 128, 'resnet_depth': 4, 'resnet_block_hidden_factor': 1.0,
            'resnet_activation_dropout': 0.1, 'resnet_residual_dropout': 0.1,
            'flow_transforms': 3, 'flow_mlp_layers_in_transform': 2, 'flow_bins': 8,
            'categorical_cardinalities': None,
        },
        "TRAIN_HYPERPARAMS": {
            'lr': 0.003, 'weight_decay': 1e-4, 'num_epochs': 400,
            'patience_early_stopping': 100, 'batch_size': 2048
        }
    },
}
# --- END: Dataset-specific configurations ---

if __name__ == '__main__':
    source = "uci"
    train_model_flag = True

    # For looping through all datasets in the source
    datasets_to_run = DATASETS.get(source, {})
    overall_results_summary = {} # To store results for all datasets
    
    test = False
    if test:
        dataset = "yacht"
        datasets_to_run = DATASETS.get(source, {}).get(dataset, None)
        print(datasets_to_run)
        if datasets_to_run:
            datasets_to_run = {dataset: datasets_to_run}
        else:
            print("Could not find a default dataset for testing. Please check DATASETS structure.")
            datasets_to_run = {}


    for dataset_key, dataset_info_dict in datasets_to_run.items():
        dataset_display_name = dataset_info_dict.get('name', dataset_key)

        MODEL_HYPERPARAMS = ALL_DATASET_MODEL_CONFIGS[dataset_key]["MODEL_HYPERPARAMS"]
        TRAIN_HYPERPARAMS = ALL_DATASET_MODEL_CONFIGS[dataset_key]["TRAIN_HYPERPARAMS"]

        if dataset_key == "protein_tertiary_structure":
            num_folds_to_run = 5
        else:
            num_folds_to_run = 2

        all_folds_test_nll = []
        all_folds_test_mae = []
        all_folds_test_mse = []
        all_folds_test_rmse = []
        all_folds_test_mape = []

        logger.info(f"===== Starting {num_folds_to_run}-Fold Evaluation for: {dataset_display_name} ({dataset_key}) =====")

        for fold_idx in range(num_folds_to_run):
            logger.info(f"--- Processing Fold {fold_idx+1}/{num_folds_to_run} for dataset: {dataset_key} ---")

            train_loader, val_loader, test_loader, _, target_scaler = \
                load_preprocessed_data(source, dataset_key, TRAIN_HYPERPARAMS['batch_size'], fold_idx)

            if not train_loader.dataset.tensors[0].numel():
                logger.warning(f"Fold {fold_idx+1} for {dataset_key} has no training data. Skipping fold.")
                # Add NaNs to ensure lists have same length for np.nanmean if other folds run
                all_folds_test_nll.append(np.nan)
                all_folds_test_mae.append(np.nan)
                all_folds_test_mse.append(np.nan)
                all_folds_test_rmse.append(np.nan)
                all_folds_test_mape.append(np.nan)
                continue

            num_numerical_features = train_loader.dataset.tensors[0].shape[1]
            effective_scale_for_density_transform = target_scaler.scale_[0]

            if train_model_flag:
                logger.info(f"Starting training process for {dataset_key}, Fold {fold_idx+1}...")
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                model_init_params = {
                    'num_numerical_features': num_numerical_features,
                    **MODEL_HYPERPARAMS,
                    'target_scaler_actual_scale': effective_scale_for_density_transform
                }
                training_model = TabResFlow(**model_init_params)
                logger.info(f"TabResFlow instantiated on {device} for {dataset_key}, Fold {fold_idx+1}")

                best_val_loss, best_model_state = training_model.fit(
                    train_loader=train_loader, val_loader=val_loader, device=device,
                    lr=TRAIN_HYPERPARAMS['lr'],
                    weight_decay=TRAIN_HYPERPARAMS['weight_decay'],
                    num_epochs=TRAIN_HYPERPARAMS['num_epochs'],
                    patience_early_stopping=TRAIN_HYPERPARAMS['patience_early_stopping'],
                    model_save_path=None,
                    dataset_key_for_save=None
                )

                logger.info(f"Evaluating on test set for {dataset_key}, Fold {fold_idx+1}...")
                test_model = TabResFlow(**model_init_params)

                if best_model_state: # If training was run and returned a state
                    test_model.load_state_dict(best_model_state)
                    logger.info(f"Loaded best model state (from validation) for test evaluation.")
                elif os.path.exists(f"trained_models/tabresflow_best_{dataset_key}.pth"): #model saved and exist , later
                    # Fallback to loading from file if training was skipped but a file exists
                    try:
                        logger.info(f"Training skipped, attempting to load model from file for testing.")
                        checkpoint = torch.load(f"trained_models/tabresflow_best_{dataset_key}.pth", map_location=device)
                        test_model.load_state_dict(checkpoint['model_state_dict'])
                        logger.info(f"Loaded model from trained_models/tabresflow_best_{dataset_key}.pth for test evaluation.")
                    except Exception as e:
                        logger.error(f"Could not load model from file: {e}. Evaluating with an untrained model state.")

                test_model.load_state_dict(best_model_state)
                test_model.to(device)
                test_model.eval()

                test_nll = evaluation.evaluate_nll(
                    model=test_model, data_loader=test_loader, device=device, current_epoch_num=TRAIN_HYPERPARAMS['num_epochs'] # Pass epoch for logging context
                )
                all_folds_test_nll.append(test_nll)
                logger.info(f"Fold {fold_idx+1} Test NLL: {test_nll:.4f}")

                regression_metrics_test = evaluation.calculate_and_log_regression_metrics_on_test(
                    model=test_model, test_loader=test_loader, device=device,
                    target_scaler=target_scaler, num_mc_samples_for_pred=1000,
                    dataset_key_for_logging=f"{dataset_key}_Fold{fold_idx+1}",
                )
                all_folds_test_mae.append(regression_metrics_test.get('MAE', np.nan))
                all_folds_test_mse.append(regression_metrics_test.get('MSE', np.nan))
                all_folds_test_rmse.append(regression_metrics_test.get('RMSE', np.nan))
                all_folds_test_mape.append(regression_metrics_test.get('MAPE', np.nan))
                logger.info(f"Fold {fold_idx+1} Test Regression Metrics: {regression_metrics_test}")
            else: # if not train_model_flag (add placeholder if you need to run without training)
                logger.info(f"Training skipped for {dataset_key}, Fold {fold_idx+1}. Appending NaNs for metrics.")
                all_folds_test_nll.append(np.nan)
                all_folds_test_mae.append(np.nan)
                all_folds_test_mse.append(np.nan)
                all_folds_test_rmse.append(np.nan)
                all_folds_test_mape.append(np.nan)


        # --- Results for the current dataset ---
        mean_nll = np.nanmean(all_folds_test_nll) if all_folds_test_nll else np.nan
        std_nll = np.nanstd(all_folds_test_nll) if all_folds_test_nll else np.nan
        mean_mse = np.nanmean(all_folds_test_mse) if all_folds_test_mse else np.nan
        std_mse = np.nanstd(all_folds_test_mse) if all_folds_test_mse else np.nan
        mean_rmse = np.nanmean(all_folds_test_rmse) if all_folds_test_rmse else np.nan
        std_rmse = np.nanstd(all_folds_test_rmse) if all_folds_test_rmse else np.nan
        mean_mae = np.nanmean(all_folds_test_mae) if all_folds_test_mae else np.nan
        std_mae = np.nanstd(all_folds_test_mae) if all_folds_test_mae else np.nan
        mean_mape = np.nanmean(all_folds_test_mape) if all_folds_test_mape else np.nan
        std_mape = np.nanstd(all_folds_test_mape) if all_folds_test_mape else np.nan

        logger.info(f"===== AGGREGATED RESULTS for {dataset_display_name} ({dataset_key}) over {num_folds_to_run} Folds =====")
        logger.info(f"Average Test NLL: {mean_nll:.4f} ± {std_nll:.4f}")
        logger.info(f"Average Test MSE: {mean_mse:.4f} ± {std_mse:.4f}")
        logger.info(f"Average Test RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
        logger.info(f"Average Test MAE: {mean_mae:.4f} ± {std_mae:.4f}")
        logger.info(f"Average Test MAPE: {mean_mape:.2f}% ± {std_mape:.2f}%")
        logger.info("===================================================================")

        overall_results_summary[dataset_key] = {
            'display_name': dataset_display_name,
            'num_folds': num_folds_to_run,
            'NLL_mean': mean_nll, 'NLL_std': std_nll,
            'MSE_mean': mean_mse, 'MSE_std': std_mse,
            'RMSE_mean': mean_rmse, 'RMSE_std': std_rmse,
            'MAE_mean': mean_mae, 'MAE_std': std_mae,
            'MAPE_mean': mean_mape, 'MAPE_std': std_mape,
        }

    # --- Final Summary for ALL Datasets ---
    logger.info("\n\n===== ***** OVERALL SUMMARY OF ALL DATASET EVALUATIONS ***** =====")
    for ds_key, results in overall_results_summary.items():
        logger.info(f"--- Dataset: {results['display_name']} ({ds_key}) ({results['num_folds']} Folds) ---")
        logger.info(f"  Ran {results['num_folds']} Folds")
        logger.info(f"  Average Test NLL: {results['NLL_mean']:.4f} ± {results['NLL_std']:.4f}")
        logger.info(f"  Average Test MSE: {results['MSE_mean']:.4f} ± {results['MSE_std']:.4f}")
        logger.info(f"  Average Test RMSE: {results['RMSE_mean']:.4f} ± {results['RMSE_std']:.4f}")
        logger.info(f"  Average Test MAE: {results['MAE_mean']:.4f} ± {results['MAE_std']:.4f}")
        logger.info(f"  Average Test MAPE: {results['MAPE_mean']:.2f}% ± {results['MAPE_std']:.2f}%")
    logger.info("===== ***** END OF OVERALL SUMMARY ***** =====")