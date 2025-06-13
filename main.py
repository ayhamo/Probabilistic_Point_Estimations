'''
TODO:

1. Copy TabResFlow openml results
2. Copy TabPFN openml results and compare
3. run xgboost on UCI with optuna , mentioned below
4. re run openml datasets with new paramters
5. update catboost, and get results both UCI and OpenML

6. implement TabResNet

X. Ask about Gaussian Process Regression and C.N.F?
X. Implement VAE
X. Implement Diffusion Models

7. ???
8. Profit.

'''

if __name__ == '__main__':

    dataset_sources = ["uci", "openml_ctr23", "multivariate"]

    optuna = False
    optuna_metrics_optimize = ["Mean NLL", "MAE", "MSE", "RMSE", "MAPE"]

    train_TabResFlow = False
    train_TabPFN = False
    train_XGBoost = False
    train_CatBoost = False
    train_TabResNet = True

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
            test_datasets = ["361253"] ,#["361622", ], # can specify a list of dataset key to test, otherwise None
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