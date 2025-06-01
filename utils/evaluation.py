from configs.logger_config import global_logger as logger
from configs.config import device

import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

from scipy.stats import gaussian_kde 
from scipy import signal # For find_peaks

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
    rmse = float(np.sqrt(mse))
    
    mask = np.abs(y_true) > 1e-6 
    if np.any(mask):
        mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100
    else:
        mape = np.nan

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape
    }

def evaluate_nll(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    current_epoch_num = 0
    ):
    """
    Evaluates the model on the given data_loader and returns the average NLL.
    """
    model.eval() # Set model to evaluation mode
    total_val_nll = 0.0
    batches_processed = 0
    with torch.no_grad():
        for batch_idx, (x_num_b_from_loader, y_b_from_loader) in enumerate(data_loader):
            if y_b_from_loader.numel() == 0: continue 
            
            x_num_b = x_num_b_from_loader.to(device)
            y_b = y_b_from_loader.to(device)
            # Create an empty x_cat_b tensor
            x_cat_b = torch.empty((x_num_b.shape[0], 0), dtype=torch.long, device=device)
                  
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

# kept for now for other models , may remove later
def evaluate_regression_test_samples(
    model: nn.Module, # Expecting model with predict_mean_std method
    test_loader: torch.utils.data.DataLoader,
    target_scaler: MinMaxScaler,
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
        for batch_idx, (x_num_b_from_loader, y_b_loader_scaled) in enumerate(test_loader):
            if y_b_loader_scaled.numel() == 0: 
                logger.debug(f"Skipping empty target batch {batch_idx} in test regression evaluation.")
                continue
            
            x_num_b = x_num_b_from_loader.to(device) if x_num_b_from_loader.numel() > 0 else x_num_b_from_loader

            # Create an empty x_cat_b tensor
            x_cat_b = torch.empty((x_num_b.shape[0], 0), dtype=torch.long, device=device)
            
            try: 

                y_pred_b_mean_scaled, y_pred_b_std_scaled = model.predict_mean_std(
                    x_num_b, x_cat_b, num_mc_samples=num_mc_samples_for_pred
                )     
                
                if y_b_loader_scaled.numel() == 0: # If predict_mean_std returns empty for an empty input
                    logger.debug(f"Skipping batch {batch_idx} due to empty predictions.")
                    continue
                
                # Inverse transform predictions and true values for metrics on original scale
                y_true_batch_scaled_np = y_b_loader_scaled.cpu().numpy().reshape(-1, 1)
                y_pred_batch_mean_scaled_np = y_pred_b_mean_scaled.cpu().numpy().reshape(-1, 1)
                y_pred_batch_std_scaled_np = y_pred_b_std_scaled.cpu().numpy().reshape(-1, 1)

                y_true_batch_original_np = target_scaler.inverse_transform(y_true_batch_scaled_np)
                y_pred_batch_mean_original_np = target_scaler.inverse_transform(y_pred_batch_mean_scaled_np)
                y_pred_batch_std_original_np = target_scaler.inverse_transform(y_pred_batch_std_scaled_np)

                all_y_true_test.extend(y_true_batch_original_np.squeeze())
                all_y_pred_test_mean.extend(y_pred_batch_mean_original_np.squeeze())
                all_y_pred_test_std.extend(y_pred_batch_std_original_np.squeeze())

            except Exception as e:
                logger.error(f"Error during prediction for regression metrics (Batch {batch_idx}): {e}", exc_info=True)
                continue
        
    try:
        regression_metrics = calculate_regression_metrics(all_y_true_test, all_y_pred_test_mean)
    except ValueError as e:
        logger.error(f"Error calculating regression metrics: {e}")
        # Optional: Handle NaNs before retrying
        if np.isnan(all_y_true_test).any() or np.isnan(all_y_pred_test_mean).any():
            logger.warning("NaN values detected in input data!")
            # for example drop?
            mask = ~np.isnan(all_y_true_test) & ~np.isnan(all_y_pred_test_mean)  # Keep only non-NaN elements
            y_true_filtered = all_y_true_test[mask]
            y_pred_filtered = all_y_pred_test_mean[mask]
            try:
                regression_metrics = calculate_regression_metrics(y_true_filtered, y_pred_filtered)
            except ValueError as e:
                print(f"Error even after NaN handling: {e}")
                regression_metrics = None  # Fallback option
        
    avg_pred_std = None
    if all_y_pred_test_std:
        y_std_np = np.array(all_y_pred_test_std)
        if y_std_np.size > 0: # Ensure not empty before mean
             avg_pred_std = float(np.mean(y_std_np))

    return regression_metrics, avg_pred_std
    
    
def calculate_and_log_regression_metrics_on_test(
    model: nn.Module, 
    test_loader: torch.utils.data.DataLoader,
    target_scaler: MinMaxScaler, 
    num_mc_samples_for_pred: int = 1000,
    dataset_key_for_logging: str = "UnknownDataset"
):
    
    logger.info(f"[{dataset_key_for_logging}] Calculating regression metrics on test set (ORIGINAL SCALE) using MODE of {num_mc_samples_for_pred} MC samples:")
    
    all_y_true_original = []
    all_y_mode_predictions_original = [] # Storing mode predictions
    
    # default for find_peaks_parameters, used in _calculate_rmse_at_k
    find_peaks_params = {"height": 0.1} 
    
    model.eval() 
    with torch.no_grad():
        for batch_idx, (x_num_b_loader, y_b_loader_scaled_targets) in enumerate(test_loader):
            if y_b_loader_scaled_targets.numel() == 0: continue
            
            x_num_b = x_num_b_loader.to(device) if x_num_b_loader.numel() > 0 else x_num_b_loader
            x_cat_b = torch.empty((x_num_b.shape[0], 0), dtype=torch.long, device=device) 
            
            # If target_scaler is provided, perform inverse scaling; otherwise, use raw values
            if target_scaler:
                true_scaled_np_batch = y_b_loader_scaled_targets.cpu().numpy().reshape(-1, 1)
                true_original_np_batch = target_scaler.inverse_transform(true_scaled_np_batch)
            else:
                true_original_np_batch = y_b_loader_scaled_targets.cpu().numpy()

            all_y_true_original.extend(true_original_np_batch.squeeze())
            
            # This method returns (batch_size, num_mc_samples) on ORIGINAL scale
            batch_samples_original = model.predict_samples_original_scale(
                x_num_b, x_cat_b, target_scaler, num_mc_samples=num_mc_samples_for_pred
            )
            if batch_samples_original.shape[0] == 0: continue

            batch_modes_original = np.array([
                get_peak_prediction(
                    batch_samples_original[i], 
                    find_peaks_parameters=find_peaks_params, 
                    n_peaks_to_return=1
                )[0] # Get the first (primary) peak
                for i in range(batch_samples_original.shape[0])
            ])
            all_y_mode_predictions_original.extend(batch_modes_original)
                

    regression_metrics = calculate_regression_metrics(
        np.array(all_y_true_original), 
        np.array(all_y_mode_predictions_original) # Use MODE predictions
    )

    return regression_metrics
    

# Adapted from authors reporting_utils.py
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

class KDE:
    """Univariate and bivariate kernel density estimator."""

    def __init__(
            self, *,
            bw_method=None,
            bw_adjust=1,
            gridsize=200,
            cut=3,
            clip=None,
            cumulative=False,
    ):
        """Initialize the estimator with its parameters.
        Parameters
        ----------
        bw_method : string, scalar, or callable, optional
            Method for determining the smoothing bandwidth to use; passed to
            :class:`scipy.stats.gaussian_kde`.
        bw_adjust : number, optional
            Factor that multiplicatively scales the value chosen using
            ``bw_method``. Increasing will make the curve smoother. See Notes.
        gridsize : int, optional
            Number of points on each dimension of the evaluation grid.
        cut : number, optional
            Factor, multiplied by the smoothing bandwidth, that determines how
            far the evaluation grid extends past the extreme datapoints. When
            set to 0, truncate the curve at the data limits.
        clip : pair of numbers or None, or a pair of such pairs
            Do not evaluate the density outside of these limits.
        cumulative : bool, optional
            If True, estimate a cumulative distribution function. Requires scipy.
        """
        if clip is None:
            clip = None, None

        self.bw_method = bw_method
        self.bw_adjust = bw_adjust
        self.gridsize = gridsize
        self.cut = cut
        self.clip = clip
        self.cumulative = cumulative
        self.support = None

    def _define_support_grid(self, x, bw, cut, clip, gridsize):
        """Create the grid of evaluation points depending for vector x."""
        clip_lo = -np.inf if clip[0] is None else clip[0]
        clip_hi = +np.inf if clip[1] is None else clip[1]
        gridmin = max(x.min() - bw * cut, clip_lo)
        gridmax = min(x.max() + bw * cut, clip_hi)
        return np.linspace(gridmin, gridmax, gridsize)

    def _define_support_univariate(self, x, weights):
        """Create a 1D grid of evaluation points."""
        kde = self._fit(x, weights)
        bw = np.sqrt(kde.covariance.squeeze())
        grid = self._define_support_grid(
            x, bw, self.cut, self.clip, self.gridsize
        )
        return grid

    def _define_support_bivariate(self, x1, x2, weights):
        """Create a 2D grid of evaluation points."""
        clip = self.clip
        if clip[0] is None or np.isscalar(clip[0]):
            clip = (clip, clip)

        kde = self._fit([x1, x2], weights)
        bw = np.sqrt(np.diag(kde.covariance).squeeze())

        grid1 = self._define_support_grid(
            x1, bw[0], self.cut, clip[0], self.gridsize
        )
        grid2 = self._define_support_grid(
            x2, bw[1], self.cut, clip[1], self.gridsize
        )
        return grid1, grid2

    def define_support(self, x1, x2=None, weights=None, cache=True): 
        """Create the evaluation grid for a given data set."""
        if x2 is None:
            support = self._define_support_univariate(x1, weights)
        else:
            support = self._define_support_bivariate(x1, x2, weights)

        if cache:
            self.support = support
        return support

    def _fit(self, fit_data, weights=None):
        """Fit the scipy kde while adding bw_adjust logic and version check."""
        fit_kws = {"bw_method": self.bw_method}
        if weights is not None:
            fit_kws["weights"] = weights

        kde = gaussian_kde(fit_data, **fit_kws)
        kde.set_bandwidth(kde.factor * self.bw_adjust)
        return kde

    def _eval_univariate(self, x, weights=None):
        """Fit and evaluate a univariate on univariate data."""
        support = self.support
        if support is None:
            support = self.define_support(x, weights=weights, cache=False) # Pass weights here

        kde = self._fit(x, weights)

        if self.cumulative:
            s_0 = support[0]
            density = np.array([
                kde.integrate_box_1d(s_0, s_i) for s_i in support
            ])
        else:
            density = kde(support)
        return density, support

    def _eval_bivariate(self, x1, x2, weights=None):
        """Fit and evaluate a univariate on bivariate data."""
        support = self.support
        if support is None:
            support = self.define_support(x1, x2, weights=weights, cache=False) # Pass weights here

        kde = self._fit([x1, x2], weights)

        if self.cumulative:
            grid1, grid2 = support
            density = np.zeros((grid1.size, grid2.size))
            p0 = grid1.min(), grid2.min()
            for i, xi in enumerate(grid1):
                for j, xj in enumerate(grid2):
                    density[i, j] = kde.integrate_box(p0, (xi, xj))
        else:
            xx1, xx2 = np.meshgrid(*support)
            density = kde([xx1.ravel(), xx2.ravel()]).reshape(xx1.shape)
        return density, support

    def __call__(self, x1, x2=None, weights=None):
        """Fit and evaluate on univariate or bivariate data."""
        if x2 is None:
            return self._eval_univariate(x1, weights)
        else:
            return self._eval_bivariate(x1, x2, weights)


def get_peak_prediction(
    sample_array: np.ndarray, # A 1D array of samples for a single data point
    find_peaks_parameters,
    n_peaks_to_return: int = 1
):
    """
    Calculates the mode (peak) from samples using the author KDE class and logic.
    """
    if sample_array.size == 0:
        return np.full(n_peaks_to_return, np.nan)
    # Author code doesn't explicitly check for sample_array.size < 2 before KDE,
    # but gaussian_kde might fail. Let's rely on its internal error handling or the try-except.

    kde_instance = KDE() # Using default parameters for KDE class as in calculate_peaks_from_sample
                        # Their KDE class defaults: bw_method=None, bw_adjust=1, gridsize=200, cut=3
    
    try:
        density, support_grid = kde_instance(sample_array) # Call KDE instance
    except Exception as e: # Catch LinAlgError or other issues if all samples are identical for KDE
        logger.debug(f"KDE computation failed for a sample array: {e}. Falling back to mean.")
        # fallback in original code is np.repeat(np.mean(sample), n_peaks)
        return np.repeat(np.mean(sample_array), n_peaks_to_return)

    peak_indices, _ = signal.find_peaks(density, **find_peaks_parameters)
    peaks_order = np.argsort(-density[peak_indices])  # Sort in descending order of density.
    peaks_on_support = support_grid[peak_indices]
    peaks_on_support_sorted = peaks_on_support[peaks_order]

    if len(peak_indices) == 0:
        # fallback: np.repeat(np.mean(sample), n_peaks)
        final_peaks = np.repeat(np.mean(sample_array), n_peaks_to_return)
    else:
        # resize logic:
        final_peaks = np.resize(peaks_on_support_sorted, n_peaks_to_return)
        # This means if fewer peaks are found than n_peaks_to_return, it repeats the found peaks.
        # If more are found, it truncates.
            
    return final_peaks

