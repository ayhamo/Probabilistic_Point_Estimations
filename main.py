'''
TODO:

1. TabResFlow openml results
2. TabPFN updated openml results and compare
3. TabPFN UCI protien update
4. run xgboost on UCI with optuna
4. re run xgboost openml and UCI datasets with new paramters
5. update catboost, and get results both UCI and OpenML
6. implement TabResNet
7. get TabResNet results
8. Implement VAE
9. get VAE results
10. get TabResFlow CRPS!

11. Gaussian Process Regression => https://arxiv.org/abs/2310.11527 Thin and Deep Gaussian Processes (NeurIPS 2023)
installation explained in requirements.txt and must be followed in order, otherwise it won't work!

12. Diffusion Models => Auto-Regressive Moving Diffusion Models for Time Series Forecasting (AAAI 2025)

13. ???
14. Profit.

'''

if __name__ == '__main__':

    dataset_sources = ["uci", "openml_ctr23", "multivariate"]

    optuna = False
    optuna_metrics_optimize = ["Mean NLL", "MAE", "MSE", "RMSE", "MAPE"]

    train_TabResFlow = False
    train_TabPFN = False
    train_XGBoost = False
    train_CatBoost = False
    train_TabResNet = False
    train_VAE = False
    train_GPs = True

    if train_TabResFlow:
        from models.TabResFlow import run_TabResFlow_pipeline, run_tabresflow_optuna

        if optuna:
            best_hyperparameters_TabResFlow = run_tabresflow_optuna(            
            source_dataset=dataset_sources[1],
            datasets_to_optimize=["361250", "361622"],
            n_trials_optuna=100, 
            hpo_fold_idx=0, 
            #metric_to_optimize= "NLL" optimizes NLL by default
        )
        else:
            TabResFlow_summary_df = run_TabResFlow_pipeline(
            source_dataset = dataset_sources[1],
            test_datasets = None ,#["361622", ], # can specify a list of dataset key to test, otherwise None
            # base_model_save_path_template="trained_models/tabresflow_best_{dataset_key}_fold{fold_idx}.pth"
        )

    if train_TabPFN:
        from models.TabPFN import run_TabPFN_pipeline
        # TabPFN does HPO automtaiclly
        TabPFN_summary_df = run_TabPFN_pipeline(
        source_dataset = dataset_sources[1], 
        test_datasets = ["361272"],
        models_train_types = ["tabpfn_regressor"] ,#  "autotabpfn_regressor"],
        # base_model_save_path_template="trained_models/tabpfn_best_{dataset_key}_fold{fold_idx}.pth"
    )
        
    if train_XGBoost:
        from models.XGBoost import run_XGBoost_pipeline, run_xgboost_optuna

        if optuna:
            best_hyperparameters_all_datasets = run_xgboost_optuna(
            source_dataset=dataset_sources[0],
            datasets_to_optimize=["concrete", "power-plant","protein-tertiary-structure"],
            n_trials_optuna=100, 
            hpo_fold_idx=1, 
            metric_to_optimize= optuna_metrics_optimize[3]
        )
        else:           
            XGBoost_summary_df = run_XGBoost_pipeline(
            source_dataset = dataset_sources[0],
            test_datasets = None,
            # base_model_save_path_template="trained_models/xgboost_best_{dataset_key}_fold{fold_idx}.pth"
            )

    if train_CatBoost:
        from models.CatBoost import run_CatBoost_pipeline

        CatBoost_summary_df = run_CatBoost_pipeline(
        source_dataset = dataset_sources[1],
        test_datasets = None,
        # base_model_save_path_template="trained_models/xgboost_best_{dataset_key}_fold{fold_idx}.pth"
        )

    if train_TabResNet:
        from models.TabResNet import run_TabResNet_pipeline

        TabResNet_summary_df = run_TabResNet_pipeline(
        source_dataset = dataset_sources[1],
        test_datasets = None,
        # base_model_save_path_template="trained_models/TabResNet_best_{dataset_key}_fold{fold_idx}.pth"
        )

    if train_VAE:
        from models.TTVAE import run_TTVAE_pipeline

        TabResNet_summary_df = run_TTVAE_pipeline(
        source_dataset = dataset_sources[1],
        test_datasets = None,
        epochs=60
        )

    if train_GPs:
        from models.TDGPs import run_TDGP_pipeline

        TDGPs_summary_df = run_TDGP_pipeline(
        source_dataset = dataset_sources[0],
        test_datasets = None,
        )