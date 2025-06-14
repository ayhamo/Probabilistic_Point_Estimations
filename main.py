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
TODO ASK!
Tab-VAE (ICML 2023 and ICPRAM 2024): Although originally designed as an unconditional VAE for generating synthetic tabular data, Tab-VAE is 
highly promising for extension into a conditional setting. Its architecture carefully tackles high-dimensional 
categorical inputs—especially those arising from one-hot encodings—by incorporating specialized inference techniques.
By simply concatenating the condition (e.g., class labels or other relevant attributes) to both the encoder and decoder
inputs, you can transform Tab-VAE into a CVAE. This extension would allow it to generate data that respects the 
underlying conditional distributions, which is particularly helpful for datasets with imbalanced classes or when 
specific data subgroups need to be generated selectively.

TTVAE (Transformer-based Tabular VAE) (EUSIPCO 2024): TTVAE leverages the attention mechanism of Transformer architectures to capture 
complex interrelationships in tabular data. While its core formulation is unconditional, its rich latent representations 
naturally lend themselves to conditioning. Integrating additional contextual tokens or concatenated condition vectors can 
refine the generation process. This adaptation makes it possible to generate synthetic data that not only reflects the 
inherent structure of the tabular domain but also adheres to external conditions—ideal for applications in finance or 
healthcare where such control is paramount.

VAE-GMM Integration Models: Another notable direction involves enhancing the VAE framework with components such as a 
Bayesian Gaussian Mixture Model (GMM) to better capture multi-modal distributions inherent in tabular datasets. While 
models like the one proposed in the improved tabular data generator with VAE-GMM integration do not start explicitly 
as conditional models, their modular design means that adding conditioning is straightforward. By conditioning the 
latent space clustering on additional attributes, one could refine the model's ability to generate feature-rich, 
condition-specific synthetic data.

9. get VAE results

TODO ASK!
C.N.F is same as NodeFlow, which TabResFlow (kiran paper) built upon, so do i really need to do them?
the same case applies to spline flows, where kiran also used them in his model

X. Ask about Gaussian Process Regression
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
    train_TabResNet = False
    train_VAE = True

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

    if train_VAE:
        from models.VAE import run_VAE_pipeline

        TabResNet_summary_df = run_VAE_pipeline(
        source_dataset = dataset_sources[1],
        test_datasets = None,
        # base_model_save_path_template="trained_models/VAE_best_{dataset_key}_fold{fold_idx}.pth"
        )