from configs.logger_config import global_logger as logger
import os
import pandas as pd
import numpy as np

'''
TODO:


1. put all ctr23 datasets, and put a table FOR ALL DATASETS train size, test size, number of input features, number of output feautres, number of catorgical features, number missing feautres, percentage missing
2. look at kiran paper new and update
3. talk to kiran and see his implemtnation and compare with folding

4. implement tabpfn, xgboost, catboost

'''

if __name__ == '__main__':

    dataset_sources = ["uci", "openml_ctr23", "multivariate"]
    train_TabResFlow = False
    train_TabPFN = False
    train_XGBoost = True

    if train_TabResFlow:
        from models.TabResFlow import run_TabResFlow_pipeline

        overall_summary_df = run_TabResFlow_pipeline(
        source_dataset = dataset_sources[0],
        test_single_dataset = "361622", # can specify a single dataset to test ie "yacht", otherwise None
        # base_model_save_path_template="trained_models/tabresflow_best_{dataset_key}_fold{fold_idx}.pth"
        )

    if train_TabPFN:
        from models.TabPFN import run_TabPFN_pipeline
        
        TabPFN_summary_df = run_TabPFN_pipeline(
        source_dataset = dataset_sources[1], 
        test_single_dataset = None, # "361622"
        models_train_types = ["tabpfn_regressor"] ,#  "autotabpfn_regressor"],
        # base_model_save_path_template="trained_models/tabpfn_best_{dataset_key}_fold{fold_idx}.pth"
    )
        
    if train_XGBoost:
        from models.XGBoost import run_XGBoost_pipeline

        XGBoost_summary_df = run_XGBoost_pipeline(
        source_dataset = dataset_sources[1],
        test_single_dataset = None, # "...."
        # base_model_save_path_template="trained_models/xgboost_best_{dataset_key}_fold{fold_idx}.pth"
        )

    #print(overall_summary_df)