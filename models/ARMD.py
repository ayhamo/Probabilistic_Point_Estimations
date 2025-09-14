import pandas as pd
import numpy as np
import torch

from configs.logger_config import global_logger as logger
from configs.config import DATASETS
from utils import evaluation
from torch.distributions import Normal
import properscoring as ps
from utils.data_loader import ARMDDataset

# Assuming these are your project's utility modules
from models.ARMDMain.engine.solver import Trainer
from models.ARMDMain.Utils.io_utils import instantiate_from_config


def initialize_ARMD_model(configs, dataloader, device):
    """
    Initializes the forecasting model from a config, builds the dataloader, and runs training.
    """
    logger.info(f"Initializing and training ARMD model...")

    model = instantiate_from_config(configs['model']).to(device)
    #model.use_ff = False
    model.fast_sampling = True
                                    # not really used
    trainer = Trainer(config=configs, args=None, model=model, dataloader={'dataloader':dataloader})
    trainer.train()
    
    return trainer


def evaluate_ARMD_model(trainer, test_dataloader, test_dataset, seq_len, num_samples):
    """
    Evaluates a trained ARMD model, calculating NLL, CRPS and regression metrics.

    """
    test_scaled = test_dataset.samples
    scaler = test_dataset.scaler
    seq_length, feat_num = seq_len*2, test_scaled.shape[-1]
    pred_length = seq_len
    real = test_scaled
    
    sample, real_ = trainer.sample_forecast_probabilistic(test_dataloader, num_samples=num_samples, shape=[seq_len, feat_num])
    mask = test_dataset.masking

    # --- 2. Inverse Transform (Un-scale) Data to Original Scale ---
    # The scaler expects data in shape (num_points, num_features).
    original_sample_shape = sample.shape
    original_real_shape = real_.shape

    # Un-scale the prediction samples
    sample_reshaped = sample.reshape(-1, feat_num)
    sample_unscaled_reshaped = scaler.inverse_transform(sample_reshaped)
    sample_unscaled = sample_unscaled_reshaped.reshape(original_sample_shape)
    
    # Un-scale the ground truth values
    real_reshaped = real_.reshape(-1, feat_num)
    real_unscaled_reshaped = scaler.inverse_transform(real_reshaped)
    real_unscaled = real_unscaled_reshaped.reshape(original_real_shape)
    
    # --- 3. Calculate NLL using PyTorch ---
    # First, convert the unscaled NumPy arrays to PyTorch Tensors
    sample_unscaled_tensor = torch.from_numpy(sample_unscaled).float()
    real_unscaled_tensor = torch.from_numpy(real_unscaled).float()

    # Now, perform calculations using PyTorch's API (with 'dim')
    mu = sample_unscaled_tensor.mean(dim=1)

    # a failsafe to replace nans
    # Check if we have enough samples to compute std
    if sample_unscaled_tensor.shape[1] < 2:
        print("not safe, doing dummy std!")
        # Not enough samples — fallback to small constant std
        sigma = torch.tensor(1e-2, device=sample_unscaled_tensor.device)
    else:
        # Safe to compute std
        sigma = sample_unscaled_tensor.std(dim=1)

    sigma = torch.clamp(sigma, min=1e-2)

    dist = Normal(loc=mu, scale=sigma)
    log_p = dist.log_prob(real_unscaled_tensor)
    avg_nll = -log_p.mean().item()

    # --- 4. Calculate CRPS using NumPy ---
    # Use the original NumPy arrays and NumPy's API
    real_np_flat = real_unscaled.flatten()
    
    num_points = real_unscaled.size # .size is the NumPy equivalent of .numel()
    num_samples = sample_unscaled.shape[1]
    # .transpose is the NumPy equivalent of .permute()
    sample_np_for_crps = sample_unscaled.transpose(0, 2, 3, 1).reshape(num_points, num_samples)

    crps_scores = ps.crps_ensemble(real_np_flat, sample_np_for_crps)
    avg_crps = crps_scores.mean()

    # --- 5. Calculate Standard Regression Metrics using NumPy ---
    y_pred_unscaled = np.median(sample_np_for_crps, axis=1)
    y_true_unscaled = real_np_flat
    
    regression_metrics = evaluation.calculate_regression_metrics(y_true_unscaled, y_pred_unscaled)
    
    logger.info(f"Test NLL: {avg_nll:.4f}, Test CRPS: {avg_crps:.4f}")
    logger.info(f"Test Regression Metrics: {regression_metrics}")
    
    return avg_nll, avg_crps, regression_metrics

def run_ARMD_pipeline(
    source_dataset: str = "openml_ctr23",
    test_datasets = None,
    base_model_save_path_template : str = None
    ):
    """
    Runs the ARMD model training and evaluation pipeline.

    Args:
        source_dataset_type (str): The source of the datasets (e.g., "openml_ctr23").
        test_datasets (list) : provide a list of dataset name configred in config.py to test model on
        base_model_save_path_template (str): A template string for saving pre-trained models

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

        # currently wave_energy cannot be run on kaggle
        if dataset_key == "361253":
            continue
        
        dataset_name = dataset_info_dict.get('name', dataset_key)
        
        if source_dataset == "uci":
            if dataset_key == "protein-tertiary-structure":
                num_folds_to_run = 5
            else:
                num_folds_to_run = 20
        elif source_dataset == "openml_ctr23":
            num_folds_to_run = 10

        # In their ablation study (Section 5.4 of the AAAI paper), turning on the gradient penalty 
        # (w_grad=True) slightly stabilizes the chain but adds ~20–30% more training time, and swapping 
        # to L2 (loss_type='l2') can lower RMSE on some datasets yet hurt MAE on outlier-rich series.
        # other params are fixed as they were found to be the best balance!

        configs = {
            'model': {
                'target': 'Models.autoregressive_diffusion.armd.ARMD',
                'params': {
                    'seq_length': 96,
                    'feature_size': 7, # will be uppdated later on each dataset
                    'timesteps': 96,
                    'sampling_timesteps': 3,
                    'loss_type': 'l1',
                    'beta_schedule': 'cosine',
                    'w_grad': False
                }
            },
            'solver': {
                'base_lr': 1.0e-3,
                'max_epochs': 2000,
                'results_folder': f'./ARMD_checkpoints/{dataset_key}',
                'gradient_accumulate_every': 2,
                'save_cycle': 1800,
                'ema': {
                    'decay': 0.995,
                    'update_interval': 10
                },
                'scheduler': {
                    'target': 'engine.lr_sch.ReduceLROnPlateauWithWarmup',
                    'params': {
                        'factor': 0.5,
                        'patience': 4000,
                        'min_lr': 1.0e-5,
                        'threshold': 1.0e-1,
                        'threshold_mode': 'rel',
                        'warmup_lr': 8.0e-4,
                        'warmup': 500,
                        'verbose': False
                    }
                }
            },
            'dataloader': {
                'batch_size': 128,
                'sample_size': 256,
                'shuffle': True
                }
            }
        
        logger.info(f"===== Starting ARMD {num_folds_to_run}-Fold Evaluation for: {dataset_name} ({dataset_key}) =====")

        dataset_fold_metrics = {'nll': [], 'mae': [], 'mse': [], 'rmse': [], 'mape': [], 'crps': []}

        for fold_idx in range(num_folds_to_run):
            logger.info(f"--- Processing Fold {fold_idx+1}/{num_folds_to_run} for dataset: {dataset_key} ---")

            gpu = "0"
            device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')

            # for training first
            window = 192 # fixed for all datasets

            jud = configs['dataloader']['shuffle']
            # to save npy processed files
            # config['dataloader']['train_dataset']['params']['output_dir'] = args.save_dir
            
            # all fixed configs below are taken from paper due to them being the sweet spot as per author
            train_dataset = ARMDDataset(
                        source=source_dataset,
                        dataset_identifier=dataset_key,
                        fold=fold_idx,
                        window=window, 
                        proportion=0.8,
                        save2npy=False, # no need to save npy file
                        neg_one_to_one=True, # scale [-1,1]
                        seed=2024,
                        period='train',
                        output_dir='./OUTPUT', # not used!
                        # not filled by author, so defualt is being used!
                        predict_length=None,
                        missing_ratio=None,
                        style='separate', 
                        distribution='geometric', 
                        mean_mask_length=3)
            
            # yacht has less than 128 length (94), so handle dynamiclly
            batch_size = min(configs['dataloader']['batch_size'], len(train_dataset))
            if batch_size < configs['dataloader']['batch_size']:
                logger.info(f"dataset has less than 128 instance, reducing to {len(train_dataset)}")

            train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=jud,
                                                    num_workers=0,
                                                    pin_memory=True,
                                                    sampler=None,
                                                    drop_last=jud)

            configs["model"]["params"]["feature_size"] = train_dataset[0].shape[-1] # to make it dynamic

            logger.info(f"Starting ARMD training for {dataset_name}, Fold {fold_idx}...")
            
            trainer = initialize_ARMD_model(configs, train_dataloader, device)

            logger.info("ARMD training finished.")


            sample_size = configs['dataloader']['sample_size']
            # not used, it's for saving npy files
            #config['dataloader']['test_dataset']['params']['output_dir'] = args.save_dir

            # won't be used, it fills missing data!
            #if args.mode == 'infill':
            #    config['dataloader']['test_dataset']['params']['missing_ratio'] = args.missing_ratio

            #elif args.mode == 'predict':
            # this is first set as this from seq_len which is 96 (they fixed it)
            pred_len = configs["model"]["params"]["seq_length"]
            
            test_dataset = ARMDDataset(
                    source=source_dataset,
                    dataset_identifier=dataset_key,
                    fold=fold_idx,
                    window=window, 
                    proportion=0.2,
                    save2npy=False, # no need to save npy file
                    neg_one_to_one=True, # scale [-1,1]
                    seed=2024,
                    period='test', # now only test will be loaded
                    output_dir='./OUTPUT', # not used!
                    # not filled by author, so defualt is being used!
                    predict_length=pred_len,
                    missing_ratio=None,
                    style='separate', 
                    distribution='geometric', 
                    mean_mask_length=3)

            test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                    batch_size=sample_size,
                                                    shuffle=False,
                                                    num_workers=0,
                                                    pin_memory=True,
                                                    sampler=None,
                                                    drop_last=False)


            logger.info(f"Evaluating ARMD on test set for {dataset_name}, Fold {fold_idx}...")

            num_samples = 30
            
            # wave_energy is skipped, i cannot get it to work on kaggle due to hardware constrains

            if dataset_key in ("361266", "361268", "361272"): #kings_country, fps_benchmark, fifa
                num_samples = 1
                            
            if dataset_key in ("361252", "361242"): #video_transcoding, superconductivity
                num_samples = 2

            #sacros, daimonds, brazilian_houses, health_insurance, physiochemical_protein    
            if dataset_key in ("361254" , "361257" , "361267", "361269", "361241"):
                num_samples = 5
                
            if dataset_key in ("361261"): #cps88wages
                num_samples = 20

            avg_nll, avg_crps, reg_metrics = evaluate_ARMD_model(
               trainer, test_dataloader, test_dataset, pred_len, num_samples # pred_len is seq_len
            )

            dataset_fold_metrics['nll'].append(avg_nll)
            dataset_fold_metrics['crps'].append(avg_crps)
            dataset_fold_metrics['mae'].append(reg_metrics.get('MAE', np.nan))
            dataset_fold_metrics['mse'].append(reg_metrics.get('MSE', np.nan))
            dataset_fold_metrics['rmse'].append(reg_metrics.get('RMSE', np.nan))
            dataset_fold_metrics['mape'].append(reg_metrics.get('MAPE', np.nan))

       
        # AGGREGATED RESULTS for the current dataset
        logger.info(f"===== AGGREGATED ARMD RESULTS for {dataset_name} ({dataset_key}) over {num_folds_to_run} Folds =====")
        
        mean_nll = np.nanmean(dataset_fold_metrics['nll'])
        std_nll = np.nanstd(dataset_fold_metrics['nll'])
        mean_crps = np.nanmean(dataset_fold_metrics['crps'])
        std_crps = np.nanstd(dataset_fold_metrics['crps'])
        mean_mse = np.nanmean(dataset_fold_metrics['mse'])
        std_mse = np.nanstd(dataset_fold_metrics['mse'])
        mean_rmse = np.nanmean(dataset_fold_metrics['rmse'])
        std_rmse = np.nanstd(dataset_fold_metrics['rmse'])
        mean_mae = np.nanmean(dataset_fold_metrics['mae'])
        std_mae = np.nanstd(dataset_fold_metrics['mae'])
        mean_mape = np.nanmean(dataset_fold_metrics['mape'])
        std_mape = np.nanstd(dataset_fold_metrics['mape'])
        
        logger.info(f"  Average Test NLL: {mean_nll:.4f} ± {std_nll:.4f}")
        logger.info(f"  Average Test CRPS: {mean_crps:.4f} ± {std_crps:.4f}")
        logger.info(f"  Average Test MSE: {mean_mse:.4f} ± {std_mse:.4f}")
        logger.info(f"  Average Test RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
        logger.info(f"  Average Test MAE: {mean_mae:.4f} ± {std_mae:.4f}")
        logger.info(f"  Average Test MAPE: {mean_mape:.2f}% ± {std_mape:.2f}%")

        overall_results_summary[dataset_key] = {
            'display_name': dataset_name, 'num_folds': num_folds_to_run,
            'NLL_mean': mean_nll, 'NLL_std': std_nll,
            'CRPS_mean': mean_crps, 'CRPS_std': std_crps,
            'MSE_mean': mean_mse, 'MSE_std': std_mse,
            'RMSE_mean': mean_rmse, 'RMSE_std': std_rmse,
            'MAE_mean': mean_mae, 'MAE_std': std_mae,
            'MAPE_mean': mean_mape, 'MAPE_std': std_mape,
        }
        logger.info("===================================================================\n")
    
    logger.info("===== ***** SUMMARY OF ALL DATASET EVALUATIONS ***** =====")
    for ds_key, results in overall_results_summary.items():
        logger.info(f"--- Dataset: {results['display_name']} ({ds_key}) ({results['num_folds']} Folds) ---")
        logger.info(f"  Average Test NLL: {results['CRPS_mean']:.4f} ± {results['CRPS_std']:.4f}")
        logger.info(f"  Average Test CRPS: {results['NLL_mean']:.4f} ± {results['NLL_std']:.4f}")
        logger.info(f"  Average Test MSE: {results['MSE_mean']:.4f} ± {results['MSE_std']:.4f}")
        logger.info(f"  Average Test RMSE: {results['RMSE_mean']:.4f} ± {results['RMSE_std']:.4f}")
        logger.info(f"  Average Test MAE: {results['MAE_mean']:.4f} ± {results['MAE_std']:.4f}")
        logger.info(f"  Average Test MAPE: {results['MAPE_mean']:.2f}% ± {results['MAPE_std']:.2f}%\n")
    logger.info("===== ***** END OF OVERALL SUMMARY ***** =====")

    results_df = pd.DataFrame.from_dict(overall_results_summary, orient='index')
    return results_df