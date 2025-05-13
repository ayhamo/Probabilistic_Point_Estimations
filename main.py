# THIS WILL BE DONE LATER FOR ALL RUNS
import pandas as pd
import os
from configs.config import (
    ALL_DATASET_IDENTIFIERS,
    POINT_ESTIMATOR_MODELS,
    # PROBABILISTIC_MODELS, # For later
    RESULTS_DIR
)
from pipeline import run_single_experiment
from configs.logger_config import global_logger as logger

def run_all_experiments():
    all_results = []

    for model_name, model_params in POINT_ESTIMATOR_MODELS.items():
        logger.info(f"===========================================")
        logger.info(f"Processing Model Type: Point Estimator - {model_name}")
        logger.info(f"===========================================")
        for dataset_id in ALL_DATASET_IDENTIFIERS:
            results = run_single_experiment(
                dataset_identifier=dataset_id,
                model_name=model_name,
                model_params=model_params,
                is_probabilistic=False,
            )
            if results:
                all_results.append(results)

    """
     # --- Probabilistic Models (later) ---
    for model_name, model_params in PROBABILISTIC_MODELS.items():
        logger.info(f"\n===========================================")
        logger.info(f"Processing Model Type: Probabilistic - {model_name}")
        logger.info(f"===========================================")
        for dataset_id in ALL_DATASET_IDENTIFIERS:
            results = run_single_experiment(
            dataset_identifier=dataset_id,
            model_name=model_name,
            model_params=model_params,
            is_probabilistic=True)
        if results:
            all_results.append(results)
    """

    # --- Save Results ---
    results_df = pd.DataFrame(all_results)
    results_filename = os.path.join(RESULTS_DIR, "experiment_metrics.csv")
    results_df.to_csv(results_filename, index=False)
    logger.info(f"All experiments complete. Results saved to {results_filename}")
    print("Final Results Summary:")
    print(results_df)


if __name__ == "__main__":    
    run_all_experiments()