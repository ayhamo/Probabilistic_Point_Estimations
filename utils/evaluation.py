from configs.logger_config import global_logger as logger

import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import torch
import torch.nn as nn


def calculate_regression_metrics(y_true, y_pred):
    """
    Calculates MAE, MSE, and MAPE.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        logger.warning("y_true or y_pred is empty in calculate_regression_metrics.")
        return {'MAE': np.nan, 'MSE': np.nan, 'MAPE': np.nan}
    

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    # might need to handle 0 case? check later
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    # look into papers with same datasets, to compare if our results change
    return {
        'MAE': mae,
        'MSE': mse,
        'MAPE': mape
    }

def evaluate_nll(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device,
    current_epoch_num = 0
    ):
    """
    Evaluates the model on the given data_loader and returns the average NLL.
    """
    model.eval() # Set model to evaluation mode
    total_val_nll = 0.0
    batches_processed = 0
    with torch.no_grad():
        for batch_idx, (x_num_b, x_cat_b, y_b) in enumerate(data_loader):
            x_num_b, x_cat_b, y_b = x_num_b.to(device), x_cat_b.to(device), y_b.to(device)
            
            try:
                # Assuming model has a log_prob method
                log_probs = model.log_prob(y_b, x_num_b, x_cat_b) 
                nll_loss = -log_probs.mean()
                
                if torch.isnan(nll_loss) or torch.isinf(nll_loss):
                    logger.warning(f"NaN or Inf NLL encountered during evaluation (Epoch {current_epoch_num}, Batch {batch_idx}). Skipping batch.")
                    continue # Skip this batch's contribution to average loss
                
                total_val_nll += nll_loss.item()
                batches_processed += 1
            except Exception as e:
                logger.error(f"Error during NLL calculation in evaluation (Epoch {current_epoch_num}, Batch {batch_idx}): {e}", exc_info=True)
                continue
                
    if batches_processed == 0:
        logger.warning(f"No batches were successfully processed during NLL evaluation (Epoch {current_epoch_num}). Returning NaN.")
        return float('nan')
        
    avg_val_nll = total_val_nll / batches_processed
    return avg_val_nll

def evaluate_regression_test_samples(
    model: nn.Module, # Expecting model with predict_mean_std method
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_mc_samples_for_pred: int = 1000):
    """
    Evaluates the model on the test set for regression metrics (MAE, MSE, RMSE, MAPE)
    and returns the average predicted standard deviation.

    Returns:
        A tuple: (regression_metrics_dict, average_predicted_std_dev)
                 Returns (None, None) if evaluation cannot be performed.
    """
    logger.info(f"Calculating regression metrics on test set using {num_mc_samples_for_pred} Monte Carlo samples for predictions:")
    all_y_true_test = []
    all_y_pred_test_mean = []
    all_y_pred_test_std = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (x_num_b, x_cat_b, y_b) in enumerate(test_loader):
            if y_b.numel() == 0: 
                logger.debug(f"Skipping empty target batch {batch_idx} in test regression evaluation.")
                continue
            
            # Only move features to device if they are not empty
            x_num_b = x_num_b.to(device) if x_num_b.numel() > 0 else x_num_b
            x_cat_b = x_cat_b.to(device) if x_cat_b.numel() > 0 else x_cat_b
            
            try:
                y_pred_b_mean, y_pred_b_std = model.predict_mean_std(
                    x_num_b, 
                    x_cat_b, 
                    num_mc_samples=num_mc_samples_for_pred
                )      
                
                if y_pred_b_mean.numel() == 0: # If predict_mean_std returns empty for an empty input
                    logger.debug(f"Skipping batch {batch_idx} due to empty predictions.")
                    continue

                all_y_true_test.extend(y_b.squeeze().cpu().numpy())
                all_y_pred_test_mean.extend(y_pred_b_mean.cpu().numpy())
                all_y_pred_test_std.extend(y_pred_b_std.cpu().numpy())

            except Exception as e:
                logger.error(f"Error during prediction for regression metrics (Batch {batch_idx}): {e}", exc_info=True)
                continue
    

    y_true_np = np.array(all_y_true_test)
    y_pred_np = np.array(all_y_pred_test_mean)
    
    if y_true_np.ndim == 0: y_true_np = y_true_np.reshape(-1) # Ensure 1D if single sample
    if y_pred_np.ndim == 0: y_pred_np = y_pred_np.reshape(-1)

        
    regression_metrics = calculate_regression_metrics(y_true_np, y_pred_np)
    
    avg_pred_std = None
    if all_y_pred_test_std:
        y_std_np = np.array(all_y_pred_test_std)
        if y_std_np.size > 0: # Ensure not empty before mean
             avg_pred_std = float(np.mean(y_std_np))

    return regression_metrics, avg_pred_std
    
def calculate_probabilistic_metrics():
    # TODO: Implement NLL, CRPS, Energy score later
    pass


