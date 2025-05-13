from configs.logger_config import global_logger as logger
import os
import numpy as np

import torch
from configs.config import DATASETS

from models.point_TabResFlow import TabResFlow
from utils.data_loader import load_preprocessed_data
import utils.evaluation as evaluation

if __name__ == '__main__':
    source = "uci"
    print_info = True 
    train_model_flag = True
    
    # For looping through all datasets in the source
    datasets_to_run = DATASETS.get(source, {})

    test = False
    if test:
        datasets_to_run = DATASETS.get(source, {}).get("wine", None)
        if datasets_to_run:
            datasets_to_run = {"wine": datasets_to_run}
        else:
            print("Could not find a default dataset for testing. Please check DATASETS structure.")
            datasets_to_run = {}

    for dataset_key, dataset_info_dict in datasets_to_run.items():
        dataset_display_name = dataset_info_dict.get('name', dataset_key)
        print(f"\n--- Processing dataset: {dataset_key} ({dataset_display_name}) ---")

        try:
            train_loader, val_loader, test_loader, num_numerical_features, dataset_name = \
                load_preprocessed_data(source, dataset_key, batch_size=64)
        except Exception as e:
            print(f"Error loading data for {dataset_key}: {e}")
            continue
            
        if print_info:
            logger.info(f"Number of numerical features: {num_numerical_features}")
            logger.info(f"Train loader batches: {len(train_loader)}, Val loader batches: {len(val_loader)}, Test loader batches: {len(test_loader)}")

        if train_model_flag:
            logger.info("Starting training process...")
            # 1. Use cuda
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # 2. Model Instantiation 
            model_params = {
                'num_numerical_features': num_numerical_features,
                'categorical_cardinalities': [], # All data is numerical, so should be empty
                'numerical_embedding_dim': 32,
                'category_embedding_dim': 8, # Kept for model structure, but not used if no cat_features
                'resnet_main_dim': 128,
                'resnet_k_multiplier': 2,
                'resnet_num_blocks': 3,
                'resnet_dropout': 0.1,
                'flow_num_transforms': 5,
                'flow_hidden_features': 128,
                'flow_num_bins': 8
            }

            model = TabResFlow(**model_params).to(device)
            logger.info(f"TabResFlow model instantiated on {device}.")

            # --- 3. Training Loop ---
            best_val_loss, best_model_state = model.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                lr=1e-4, #5e-4
                weight_decay=1e-5,
                num_epochs=2,
                patience_early_stopping=15,
                model_save_path= None, # "trained_models/tabresflow_best.pth"
                dataset_key_for_save=dataset_key
            )

            logger.info(f"Evaluating on test set for {dataset_key}...")

            # new model for testing
            test_model = TabResFlow(**model_params)

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

            test_model.to(device)
            
            test_nll = evaluation.evaluate_nll(
                model=test_model, data_loader=test_loader, device=device
            )
            logger.info(f"Test Set Negative Log likelihood (NLL) for {dataset_key}: {test_nll:.4f}")

            regression_metrics_test, avg_pred_std_test = evaluation.evaluate_regression_test_samples(
                model=test_model,
                test_loader=test_loader,
                device=device,
                num_mc_samples_for_pred=1000 
            )
            logger.info(f"Test Set Regression Metrics for {dataset_key}: {regression_metrics_test}")
            
            # Optionally log average predicted standard deviation
            if avg_pred_std_test:
                avg_pred_std = np.mean(avg_pred_std_test)
                logger.info(f"Test Set Average Predicted Std Dev for {dataset_key}: {avg_pred_std:.4f}")
            else:
                logger.warning(f"Could not compute regression metrics for {dataset_key} due to empty true/pred lists.")