from configs.logger_config import global_logger as logger

from configs.config import DATASETS, RANDOM_STATE, DATASET_MODEL_CONFIGS, device
from utils.data_loader import load_preprocessed_data
import utils.evaluation as evaluation

import copy
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim
import zuko

# Numerical Embedding MLP
class NumericalFeatureEncoder(nn.Module):
    def __init__(self, output_embedding_dim: int, intermediate_dim: int = 100):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, output_embedding_dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

# ResNet Block Helper
class ResNetBlock(nn.Module):
    def __init__(self, 
                 input_dim: int, # This is 'd' (resnet_main_processing_dim)
                 hidden_layer_multiplier: float = 1.0,
                 activation_dropout_rate: float = 0.1,
                 residual_dropout_rate: float = 0.1):
        super().__init__()
        actual_hidden_dim = int(input_dim * hidden_layer_multiplier) 
        
        self.bn = nn.BatchNorm1d(input_dim)
        self.linear1 = nn.Linear(input_dim, actual_hidden_dim)

        self.activation = nn.ReLU() 
        self.dropout_after_activation = nn.Dropout(activation_dropout_rate)
        self.linear2 = nn.Linear(actual_hidden_dim, input_dim)
        self.dropout_before_residual_add = nn.Dropout(residual_dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.bn(x)
        out = self.linear1(out) # Corrected: operate on 'out' from bn
        out = self.activation(out)
        out = self.dropout_after_activation(out)
        out = self.linear2(out)
        out = self.dropout_before_residual_add(out)
        out += identity 
        return out

class TabResFlow(nn.Module):
    def __init__(
        self,
        num_numerical_features: int,
        
        # Numerical Feature Embedding Specific:
        embedding_dim_per_feature: int = 64,     # Author 'dim' for ResNetModel individual feature embeddings
        numerical_encoder_intermediate_dim: int = 100, # For the 2-layer numerical MLP encoder

        # ResNet Backbone Specific:
        resnet_main_processing_dim: int = 256, # Author 'hidden_dim' (d) in ResNetModel; context dim for flow
        resnet_depth: int = 4,                 # Author 'depth' (number of ResNet blocks)
        resnet_block_hidden_factor: float = 1.0, # Author 'd_hidden_factor' for MLP within ResNet block
                                                 # Multiplies resnet_main_processing_dim for hidden layer in block
        resnet_activation_dropout: float = 0.1,  # Author 'hidden_dropout' (after activation in block)
        resnet_residual_dropout: float = 0.1,    # Author 'residual_dropout' (before adding residual)
        
        # Normalizing Flow Head Specific:
        flow_transforms: int = 3,                       # Author 'flow_num_blocks' for NSF
        flow_mlp_layers_in_transform: int = 3,          # Author 'flow_layers' (num hidden layers in each NSF transform's MLP)
        # flow_mlp_hidden_features_in_transform is implicitly resnet_main_processing_dim (see NSF init below)
        flow_bins: int = 8,                           
        
        # Categorical params (currently unused as UCI is numerical)
        categorical_cardinalities = None, 
        category_embedding_dim: int = 64, # Usually same as embedding_dim_per_feature
        target_scaler_actual_scale = None,
    ):
        super().__init__()
        self.num_numerical = num_numerical_features
        self.num_categorical = len(categorical_cardinalities) if categorical_cardinalities else 0

        # 1. Numerical Feature Encoders
        self.numerical_encoders = nn.ModuleList([
            NumericalFeatureEncoder(
                output_embedding_dim=embedding_dim_per_feature,
                intermediate_dim=numerical_encoder_intermediate_dim
            ) for _ in range(self.num_numerical)
        ])

        # 2. Categorical Feature Embeddings (if any)
        self.categorical_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=card, embedding_dim=category_embedding_dim)
            for card in (categorical_cardinalities if categorical_cardinalities else [])
        ])

        # 3. ResNet Backbone Construction
        total_concatenated_embedding_dim = (self.num_numerical * embedding_dim_per_feature) + \
                                     (self.num_categorical * category_embedding_dim)
        
        if total_concatenated_embedding_dim == 0:
            raise ValueError("Model initialized with no features, leading to zero embedding dimension.")

        # Initial projection layer
        self.input_projection = nn.Linear(total_concatenated_embedding_dim, resnet_main_processing_dim)
        
        # ResNet Blocks
        self.resnet_blocks = nn.ModuleList([
            ResNetBlock(
                input_dim=resnet_main_processing_dim,
                hidden_layer_multiplier=resnet_block_hidden_factor,
                activation_dropout_rate=resnet_activation_dropout,
                residual_dropout_rate=resnet_residual_dropout
            ) for _ in range(resnet_depth)
        ])

        # 4. Conditional Normalizing Flow Head
        # Aligning with author NSF initialization:
        # hidden_features for NSF transform's MLP = [context_dim] * num_layers_in_transform_mlp
        nsf_transform_mlp_hidden_features = [resnet_main_processing_dim] * flow_mlp_layers_in_transform

        self.normalizing_flow = zuko.flows.NSF(
            features=1, # Univariate target                          
            context=resnet_main_processing_dim, 
            transforms=flow_transforms,
            bins=flow_bins,
            hidden_features=nsf_transform_mlp_hidden_features, 
            randperm=False, 
        )

        self.target_scaler_actual_scale = target_scaler_actual_scale

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor):
        num_embeds = []
        if self.num_numerical > 0 and x_num.numel() > 0:
            # x_num shape: (batch_size, num_numerical_features)
            for i in range(self.num_numerical):
                feat = x_num[:, i:i+1] # Input to NumericalFeatureEncoder is (batch_size, 1)
                embed = self.numerical_encoders[i](feat) # Output: (batch_size, embedding_dim_per_feature)
                num_embeds.append(embed)

        cat_embeds = []
        if self.num_categorical > 0 and x_cat.numel() > 0:
            for i, emb_layer in enumerate(self.categorical_embeddings):
                cat_embeds.append(emb_layer(x_cat[:, i])) # x_cat[:, i] is (batch_size,)
        
        all_embeds = []

        if num_embeds: all_embeds.extend(num_embeds) # List of tensors
        if cat_embeds: all_embeds.extend(cat_embeds) # List of tensors
        
        if not all_embeds:
            batch_s = x_num.shape[0] if x_num.numel() > 0 else (x_cat.shape[0] if x_cat.numel() > 0 else 0)
            if batch_s == 0:
                 return torch.empty(0, self.normalizing_flow.context_size, device=next(self.parameters()).device)
            # This case should ideally be caught by __init__ if no features are defined at all.
            logger.error("No embeddings generated in forward pass for a non-empty batch. Check model init and input data.")
            # Fallback to zero context, but this is problematic.
            return torch.zeros(batch_s, self.normalizing_flow.context_size, device=next(self.parameters()).device)

        combined_embeddings = torch.cat(all_embeds, dim=1) # Concatenate all feature embeddings
        
        z = self.input_projection(combined_embeddings)
        for block in self.resnet_blocks:
            z = block(z)
        return z

    def log_prob(self, y: torch.Tensor, x_num: torch.Tensor, x_cat: torch.Tensor):
        if y.numel() == 0: return torch.empty(0, device=y.device)
        context = self.forward(x_num, x_cat)
        if context.shape[0] == 0: return torch.empty(0, device=context.device)
        
        distribution = self.normalizing_flow(context)
        log_p_y_scaled = distribution.log_prob(y) # y is the scaled target batch

        # correction is essential when the target variable has been scaled—ensuring that the computed negative log-likelihood (NLL) refers to the original scale.
        log_s_eff = torch.log(torch.tensor(self.target_scaler_actual_scale, device=y.device, dtype=y.dtype))
        log_p_original_scale = log_p_y_scaled + log_s_eff
        return log_p_original_scale
 

    def sample(self, x_num: torch.Tensor, x_cat: torch.Tensor, num_samples: int = 100):
        if x_num.numel() == 0 and x_cat.numel() == 0 and self.num_numerical == 0 and self.num_categorical == 0 : # Truly no input and no features defined
            # This case should ideally be prevented by __init__ checks for num_features > 0
            return torch.empty(0, num_samples, 1, device=next(self.parameters()).device)
        
        context = self.forward(x_num, x_cat)
        if context.shape[0] == 0: # If forward returned empty context for an empty batch input
             return torch.empty(0, num_samples, 1, device=context.device)
             
        distribution = self.normalizing_flow(context)
        # rsample returns (S, B, D) where S=num_samples, B=batch_size, D=feature_dim (1)
        return distribution.rsample((num_samples,)) 

    def predict_mean_std(self, x_num: torch.Tensor, x_cat: torch.Tensor, num_mc_samples: int = 500):
        self.eval() 
        # Handle case where input batch might be empty (e.g., last batch from DataLoader)
        current_batch_size = x_num.shape[0] if self.num_numerical > 0 and x_num.numel() > 0 else (x_cat.shape[0] if self.num_categorical > 0 and x_cat.numel() > 0 else 0)
        if current_batch_size == 0:
            return torch.empty(0, device=device), torch.empty(0, device=device)

        with torch.no_grad():
            samples = self.sample(x_num, x_cat, num_samples=num_mc_samples) # Shape: (num_mc_samples, batch_size, 1)
            
            # Check if sampling itself returned empty due to an empty context from forward()
            if samples.numel() == 0 or samples.shape[1] == 0: # samples.shape[1] is batch_size
                model_device = next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device('cpu')
                return torch.empty(0, device=model_device), torch.empty(0, device=model_device)
            
            mean = samples.mean(dim=0) # Shape: (batch_size, 1)
            std = samples.std(dim=0)   # Shape: (batch_size, 1)
        return mean.squeeze(-1), std.squeeze(-1) # Shape: (batch_size,)
    
    def predict_samples_original_scale(
        self, 
        x_num: torch.Tensor, 
        x_cat: torch.Tensor, 
        target_scaler: MinMaxScaler,
        num_mc_samples: int = 1000
    ):
        """
        Generates samples from the predictive distribution and returns them
        on the original data scale.
        Output shape: (batch_size, num_mc_samples)
        """
        self.eval()
        current_batch_size = x_num.shape[0] if self.num_numerical > 0 and x_num.numel() > 0 else \
                                (x_cat.shape[0] if self.num_categorical > 0 and x_cat.numel() > 0 else 0)
        
        if current_batch_size == 0:
            return np.array([]).reshape(0, num_mc_samples) # Return empty array with correct second dim

        with torch.no_grad():
            # self.sample returns (num_mc_samples, batch_size, 1) in scaled space
            samples_scaled = self.sample(x_num, x_cat, num_samples=num_mc_samples)
            
            if samples_scaled.numel() == 0 or samples_scaled.shape[1] == 0: # samples.shape[1] is batch_size
                    return np.array([]).reshape(0, num_mc_samples)

            # Reshape for inverse_transform: (num_mc_samples * batch_size, 1)
            batch_size_actual = samples_scaled.shape[1] # Get actual batch size from samples
            samples_scaled_reshaped = samples_scaled.permute(1, 0, 2).reshape(batch_size_actual * num_mc_samples, 1)
            samples_scaled_np = samples_scaled_reshaped.cpu().numpy()

            if target_scaler is None:
                logger.error("Target scaler is None in predict_samples_original_scale. Cannot inverse transform.")
                # Or raise error, or return scaled samples with a warning
                return samples_scaled.permute(1,0,2).squeeze(-1).cpu().numpy() # (batch_size, num_mc_samples)

            samples_original_np_flat = target_scaler.inverse_transform(samples_scaled_np)
            
            # Reshape back to (batch_size, num_mc_samples)
            samples_original_np = samples_original_np_flat.reshape(batch_size_actual, num_mc_samples)
            
        return samples_original_np

    def fit(self,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            lr: float = 1e-4, 
            weight_decay: float = 1e-5, 
            num_epochs: int = 100, 
            patience_early_stopping: int = 15,
            model_save_path = None,
            dataset_key_for_save = None
           ):

        self.to(device)
        optimizer = optim.RAdam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        best_val_loss = float('inf')
        best_model_state_dict = None 
        epochs_no_improve = 0

        logger.info(f"Starting training on {device} for {num_epochs} epochs... LR={lr}, WD={weight_decay}, Optim: RAdam")
        
        for epoch in range(num_epochs):
            self.train()
            total_train_loss = 0.0
            batches_processed_train = 0
            for batch_idx, (x_num_b_loader, y_b_loader) in enumerate(train_loader): # Expecting 2 items
                if y_b_loader.numel() == 0 : continue 
                
                x_num_b = x_num_b_loader.to(device)
                y_b = y_b_loader.to(device)
                x_cat_b = torch.empty((x_num_b.shape[0], 0), dtype=torch.long, device=device)

                optimizer.zero_grad()
                try:
                    log_probs = self.log_prob(y_b, x_num_b, x_cat_b)
                    if log_probs.numel() == 0 : continue 
                    loss = -log_probs.mean()
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning(f"NaN or Inf loss in training: epoch {epoch+1}, batch {batch_idx}. Skipping batch.")
                        continue
                    loss.backward()
                    optimizer.step()
                    total_train_loss += loss.item()
                    batches_processed_train +=1
                except Exception as e:
                    logger.error(f"Error during training batch (Epoch {epoch+1}, Batch {batch_idx}): {e}", exc_info=True)
                    continue
            
            avg_train_loss = total_train_loss / batches_processed_train if batches_processed_train > 0 else float('nan')

            avg_val_loss = evaluation.evaluate_nll(
                model=self, data_loader=val_loader, current_epoch_num=epoch+1
            )
            
            if not np.isnan(avg_val_loss) and avg_val_loss < best_val_loss:

                logger.info(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
                best_val_loss = avg_val_loss
                best_model_state_dict = copy.deepcopy(self.state_dict()) 
                epochs_no_improve = 0

                if model_save_path : 
                    final_save_path = model_save_path
                    if dataset_key_for_save:
                         final_save_path = model_save_path.replace(".pth", f"_{dataset_key_for_save}.pth")
                    save_dir = os.path.dirname(final_save_path)
                    if save_dir and not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)
                    torch.save({
                        'epoch': epoch, 'model_state_dict': best_model_state_dict,
                        'optimizer_state_dict': optimizer.state_dict(), 'loss': best_val_loss,
                    }, final_save_path)
                    logger.info(f"Model state also saved to {final_save_path}")
            elif not np.isnan(avg_val_loss): 
                epochs_no_improve += 1
            
            if epochs_no_improve >= patience_early_stopping:
                logger.info(f"Early stopping at epoch {epoch+1} after {patience_early_stopping} epochs with no improvement.")
                break
        
        logger.info(f"Training finished. Best Validation NLL: {best_val_loss:.4f}")
        return best_val_loss, best_model_state_dict
    

def run_TabResFlow_pipeline(
    source_dataset :str = "uci",
    test_datasets = None,
    base_model_save_path_template : str = None # "trained_models/tabresflow_best_{dataset_key}.pth"
    ):
    """
    Runs the TabResFlow model training and evaluation pipeline.

    Args:
        source_dataset_type (str): The source of the datasets (e.g., "uci").
        test_datasets (list) : provide a list of dataset name configred in config.py to test model on
        base_model_save_path_template (str): A template string for loading pre-trained models
                                                      Example: "trained_models/tabresflow_best_{dataset_key}.pth"

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
            MODEL_HYPERPARAMS = DATASET_MODEL_CONFIGS[dataset_key]["TabResFlow_MODEL_HYPERPARAMS"]
            TRAIN_HYPERPARAMS = DATASET_MODEL_CONFIGS[dataset_key]["TabResFlow_TRAIN_HYPERPARAMS"]

            if dataset_key == "protein-tertiary-structure":
                num_folds_to_run = 5
            else:
                num_folds_to_run = 20
        elif source_dataset == "openml_ctr23":
            num_folds_to_run = 1#20
            MODEL_HYPERPARAMS = DATASET_MODEL_CONFIGS["openML-general"]["TabResFlow_MODEL_HYPERPARAMS"]
            TRAIN_HYPERPARAMS = DATASET_MODEL_CONFIGS["openML-general"]["TabResFlow_TRAIN_HYPERPARAMS"]
 

        all_folds_test_nll = []
        all_folds_test_mae = []
        all_folds_test_mse = []
        all_folds_test_rmse = []
        all_folds_test_mape = []

        logger.info(f"===== Starting TabResFlow {num_folds_to_run}-Fold Evaluation for: {dataset_name} ({dataset_key}) =====\n")

        for fold_idx in range(num_folds_to_run):
            logger.info(f"--- Processing Fold {fold_idx+1}/{num_folds_to_run} for dataset: {dataset_key} ---")

            
            train_loader, val_loader, test_loader, target_scaler = \
                load_preprocessed_data("TabResFlow", source_dataset, dataset_key, fold_idx,
                        batch_size=TRAIN_HYPERPARAMS['batch_size'], openml_pre_prcoess=True)


            num_numerical_features = train_loader.dataset.tensors[0].shape[1]
            effective_scale_for_density_transform = target_scaler.scale_[0] if target_scaler is not None else None

            logger.info(f"Starting training process for {dataset_key}, Fold {fold_idx+1}...")

            model_init_params = {
                'num_numerical_features': num_numerical_features,
                **MODEL_HYPERPARAMS,
                'target_scaler_actual_scale': effective_scale_for_density_transform
            }
            training_model = TabResFlow(**model_init_params)
            logger.info(f"TabResFlow instantiated on {device} for {dataset_key}, Fold {fold_idx+1}")

            best_val_loss, best_model_state = training_model.fit(
                train_loader=train_loader, val_loader=val_loader,
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
                model=test_model, data_loader=test_loader, current_epoch_num=TRAIN_HYPERPARAMS['num_epochs'] # Pass epoch for logging context
            )
            all_folds_test_nll.append(test_nll)
            logger.info(f"Fold {fold_idx+1} Test NLL: {test_nll:.4f}")

            regression_metrics_test = evaluation.calculate_and_log_regression_metrics_on_test(
                model=test_model, test_loader=test_loader,
                target_scaler=target_scaler, num_mc_samples_for_pred=1000,
                dataset_key_for_logging=f"{dataset_key}_Fold {fold_idx+1}",
            )
            all_folds_test_mae.append(regression_metrics_test.get('MAE', np.nan))
            all_folds_test_mse.append(regression_metrics_test.get('MSE', np.nan))
            all_folds_test_rmse.append(regression_metrics_test.get('RMSE', np.nan))
            all_folds_test_mape.append(regression_metrics_test.get('MAPE', np.nan))
            logger.info(f"Fold {fold_idx+1} Test Regression Metrics using 100 MC samples: {regression_metrics_test}\n")


        # Results for the current dataset
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

        logger.info(f"===== AGGREGATED RESULTS for {dataset_name} ({dataset_key}) over {num_folds_to_run} Folds =====")
        logger.info(f"Average Test NLL: {mean_nll:.4f} ± {std_nll:.4f}")
        logger.info(f"Average Test MSE: {mean_mse:.4f} ± {std_mse:.4f}")
        logger.info(f"Average Test RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
        logger.info(f"Average Test MAE: {mean_mae:.4f} ± {std_mae:.4f}")
        logger.info(f"Average Test MAPE: {mean_mape:.2f}% ± {std_mape:.2f}%")
        logger.info("===================================================================\n")

        overall_results_summary[dataset_key] = {
            'display_name': dataset_name,
            'num_folds': num_folds_to_run,
            'NLL_mean': mean_nll, 'NLL_std': std_nll,
            'MSE_mean': mean_mse, 'MSE_std': std_mse,
            'RMSE_mean': mean_rmse, 'RMSE_std': std_rmse,
            'MAE_mean': mean_mae, 'MAE_std': std_mae,
            'MAPE_mean': mean_mape, 'MAPE_std': std_mape,
        }

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

    return pd.DataFrame.from_dict(overall_results_summary, orient='index')


def run_tabresflow_optuna(
    source_dataset: str,
    datasets_to_optimize: list,
    n_trials_optuna: int = 3,
    hpo_fold_idx: int = 0,
    default_batch_size_for_hpo_data_loading: int = 1024 # Initial BS for loading data
):
    """
    Runs Optuna hyperparameter optimization for TabResFlow on specified datasets.
    Hyperparameters are defined and tuned within this function.

    Args:
        source_dataset (str): The source of the datasets.
        datasets_to_optimize (list): A list of dataset keys (strings) for HPO.
        n_trials_optuna (int): Number of Optuna trials.
        hpo_fold_idx (int): Fold index for HPO data.
        default_batch_size_for_hpo_data_loading (int): Batch size used to load the HPO
                                                       dataset initially. If batch_size is
                                                       tuned by Optuna, DataLoaders will
                                                       be re-created in the objective.
    Returns:
        dict: Best hyperparameters and metric value per dataset.
    """

    import optuna

    all_best_hyperparams = {}

    for dataset_key in datasets_to_optimize:
        logger.info(f"===== Starting Optuna HPO for TabResFlow on dataset: {dataset_key}, Fold: {hpo_fold_idx+1} =====")

        # this is loaded at first for global training
        train_loader_hpo_base, val_loader_hpo_base, _, target_scaler_hpo = \
            load_preprocessed_data("TabResFlow", source_dataset, dataset_key, hpo_fold_idx,
                                   batch_size=default_batch_size_for_hpo_data_loading,
                                   openml_pre_prcoess=True)

        num_numerical_features = train_loader_hpo_base.dataset.tensors[0].shape[1]
        effective_scale_for_density_transform = target_scaler_hpo.scale_[0] if target_scaler_hpo is not None else 1.0

        # Default Fixed Model Hyperparameters (not tuned by Optuna in this setup)
        default_fixed_model_hps = {
            'numerical_encoder_intermediate_dim': 100,
            'resnet_block_hidden_factor': 1.0,
            'categorical_cardinalities': [],  
        }

        def objective(trial: optuna.trial.Trial):
            if device.type == 'cuda': torch.cuda.empty_cache()

            current_model_hps = copy.deepcopy(default_fixed_model_hps)
            
            # Model Hyperparameters to tune (mapped from paper's list)
            current_model_hps['resnet_main_processing_dim'] = trial.suggest_int('resnet_main_processing_dim', 32, 1024, log=True) 
            current_model_hps['embedding_dim_per_feature'] = trial.suggest_int('embedding_dim_per_feature', 32, 1024, log=True)
            current_model_hps['resnet_depth'] = trial.suggest_int('resnet_depth', 1, 20)
            current_model_hps['resnet_activation_dropout'] = trial.suggest_float('resnet_activation_dropout', 0.0, 0.5)
            current_model_hps['resnet_residual_dropout'] = trial.suggest_float('resnet_residual_dropout', 0.0, 0.5)
            current_model_hps['flow_transforms'] = trial.suggest_int('flow_transforms', 1, 20)
            current_model_hps['flow_bins'] = trial.suggest_int('flow_bins', 6, 20)
            current_model_hps['flow_mlp_layers_in_transform'] = trial.suggest_int('flow_mlp_layers_in_transform', 1, 12)


            # Training Hyperparameters to tune
            current_train_hps = {}
            current_train_hps['lr'] = trial.suggest_float('lr', 1e-5, 1e-2, log=True)

            current_train_hps['batch_size'] = trial.suggest_categorical('batch_size', [1024, 2048])
            current_train_hps['weight_decay'] = 1e-4

            if source_dataset != "openml_ctr23":
                current_train_hps['hpo_max_epochs'] = DATASET_MODEL_CONFIGS[dataset_key]["TabResFlow_TRAIN_HYPERPARAMS"].get('num_epochs', 400)
                current_train_hps['hpo_patience'] = DATASET_MODEL_CONFIGS[dataset_key]["TabResFlow_TRAIN_HYPERPARAMS"].get('patience_early_stopping', 400) / 4
            else:
                current_train_hps['hpo_max_epochs'] = 400
                current_train_hps['hpo_patience'] = 100

            # Prepare DataLoaders if batch_size is tuned
            train_loader_trial = train_loader_hpo_base
            val_loader_trial = val_loader_hpo_base

            # if we change the batch size, then reload the data, else we use the global one!
            if current_train_hps['batch_size'] != default_batch_size_for_hpo_data_loading:
                logger.debug(f"Recreating DataLoaders for batch_size {current_train_hps['batch_size']}")
                train_loader_trial, val_loader_trial, _, _ = \
                    load_preprocessed_data("TabResFlow", source_dataset, dataset_key, hpo_fold_idx,
                                            batch_size=current_train_hps['batch_size'],
                                            openml_pre_prcoess=True)

            model_init_params = {
                'num_numerical_features': num_numerical_features,
                **current_model_hps,
                'target_scaler_actual_scale': effective_scale_for_density_transform,
            }
            
            model = TabResFlow(**model_init_params)
            
            val_metric_value, _ = model.fit(
                train_loader=train_loader_trial,
                val_loader=val_loader_trial,
                lr=current_train_hps['lr'],
                weight_decay=current_train_hps['weight_decay'],
                num_epochs=current_train_hps['hpo_max_epochs'],
                patience_early_stopping=current_train_hps['hpo_patience'],
                model_save_path=None,
                dataset_key_for_save=None
            )

            if np.isnan(val_metric_value) or np.isinf(val_metric_value):
                logger.warning(f"Trial {trial.number} for {dataset_key} resulted in NaN/Inf metric: {val_metric_value}. Pruning or returning high value.")
                return float('inf') 

            return val_metric_value # NLL in this case always
            

        study = optuna.create_study(direction='minimize')
        
        optuna_original_verbosity = optuna.logging.get_verbosity()
        optuna.logging.set_verbosity(optuna.logging.WARNING)


        study.optimize(objective, n_trials=n_trials_optuna,
                        callbacks=[lambda st, tr: torch.cuda.empty_cache() if device.type == 'cuda' else None])

        
        optuna.logging.set_verbosity(optuna_original_verbosity)

        logger.info(f"  Optuna HPO finished for {dataset_key}.")
        logger.info(f"  Best trial number: {study.best_trial.number}")
        logger.info(f"  Best Validation NLL: {study.best_value:.4f}")
        
        best_hp_subset_optuna = study.best_params # Only contains HPs Optuna tuned
        
        # Construct the full set of best parameters
        final_best_model_hps = copy.deepcopy(default_fixed_model_hps)
        final_best_train_hps = {} # Start fresh for train HPs

        for hp_name, hp_value in best_hp_subset_optuna.items():
            # Check against keys used in trial.suggest_...
            if hp_name in ['resnet_main_processing_dim', 'embedding_dim_per_feature', 'resnet_depth',
                           'resnet_activation_dropout', 'resnet_residual_dropout', 'flow_transforms']:
                final_best_model_hps[hp_name] = hp_value
            elif hp_name in ['lr', 'batch_size', 'weight_decay']:
                final_best_train_hps[hp_name] = hp_value
        
        # Add non-tuned but essential train params
        final_best_train_hps['num_epochs'] = 400
        final_best_train_hps['patience_early_stopping'] =  100

        logger.info(f"  Best Suggested Model Hyperparameters: {final_best_model_hps}")
        logger.info(f"  Best Suggested Training Hyperparameters: {final_best_train_hps}")

        all_best_hyperparams[dataset_key] = {
            'best_model_params': final_best_model_hps,
            'best_train_params': final_best_train_hps,
            'best_value': study.best_value,
        }
        logger.info("===================================================================\n")
        if device.type == 'cuda': torch.cuda.empty_cache()


    logger.info("===== ***** SUMMARY OF OPTUNA HPO RESULTS ***** =====")
    for ds_key, result in all_best_hyperparams.items():
        dataset_name = DATASETS.get(source_dataset, {}).get(ds_key, {}).get('name', ds_key)
        logger.info(f"--- Dataset: {dataset_name} ({ds_key}) ---")
        logger.info(f"  Best NLL: {result['best_value']:.4f}")
        logger.info(f"  Best Train Hyperparameters: {result['best_train_params']}\n")
        logger.info(f"  Best Model parameters: {result['best_model_params']}\n")
    logger.info("===== ***** END OF OPTUNA HPO SUMMARY ***** =====")
    
    return all_best_hyperparams