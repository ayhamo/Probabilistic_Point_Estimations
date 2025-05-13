import os
import time
import pandas as pd

from configs.logger_config import global_logger as logger

from models.point_TabResFlow import PointEstimator
from utils.data_loader import load_preprocessed_data
from utils.evaluation import calculate_regression_metrics
from configs.config import RANDOM_STATE

def run_single_experiment(dataset_identifier, model_name, model_params=None, is_probabilistic=False):
    logger.info(f"--- Running Experiment ---")
    logger.info(f"Dataset: {dataset_identifier}, Model: {model_name}")

    # 1. Load Data
    data_package = load_preprocessed_data(dataset_identifier)
    if data_package is None:
        logger.warning(f"Failed to load data for {dataset_identifier}. Skipping experiment.")
        return None
    X_train, X_test, y_train, y_test, _ = data_package
        
    # 2. Initialize Model
    if is_probabilistic:
        # not yet!
        logger.warning(f"Probabilistic model '{model_name}' not yet implemented in this path.")
        return None
    else:
        model_instance = PointEstimator(model_type=model_name, model_params=model_params, random_state=RANDOM_STATE)

    # 3. Train Model
    start_train_time = time.time()
    try:
        model_instance.train(X_train, y_train)
    except ValueError as e:
        logger.warning(f"Error training {model_name} on {dataset_identifier}: {e}")
        return {
            'dataset': dataset_identifier,
            'model': model_name,
            'MAE': float('nan'),
            'MSE': float('nan'),
            'MAPE': float('nan'),
            'training_time_s': time.time() - start_train_time,
            'inference_time_s': 0,
            'status': f'Training Error: {e}'
        }
    training_time_s = time.time() - start_train_time
    logger.info(f"Training completed in {training_time_s:.2f}s")

    # 4. Predict
    start_infer_time = time.time()
    predictions = model_instance.predict(X_test)
    inference_time_s = time.time() - start_infer_time
    logger.info(f"Inference completed in {inference_time_s:.2f}s")

    if predictions is None or len(predictions) != len(y_test):
        print(f"Prediction failed or returned unexpected number of results for {model_name} on {dataset_identifier}.")
        return {
            'dataset': dataset_identifier,
            'model': model_name,
            'MAE': float('nan'),
            'MSE': float('nan'),
            'MAPE': float('nan'),
            'training_time_s': training_time_s,
            'inference_time_s': inference_time_s,
            'status': 'Prediction Error'
        }

    # 5. Evaluate
    metrics = calculate_regression_metrics(y_test, predictions)
    
    results = {
        'dataset': dataset_identifier,
        'model': model_name,
        **metrics, # Unpack MAE, MSE, MAPE
        'training_time_s': round(training_time_s, 4),
        'inference_time_s': round(inference_time_s, 4),
        'status': 'Success'
    }
    
    logger.info(f"Results for {model_name} on {dataset_identifier}: {metrics}")
    return results

# run single model
if __name__ == '__main__':

    experiment_result = run_single_experiment(
        dataset_identifier="openml_Buzzinsocialmedia_Twitter",
        model_name="LinearRegression",
        model_params={},
    )
    #if experiment_result:
    #    logger.debug("--- Single Experiment Result ---")
    #    for key, value in experiment_result.items():
    #        logger.debug(f"{key}: {value}")

    # Test with a dataset that might not exist to check error handling
    logger.debug("--- Testing with non-existent dataset ---")
    run_single_experiment("error_dataset", "LinearRegression")