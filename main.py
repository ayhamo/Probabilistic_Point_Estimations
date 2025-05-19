from configs.logger_config import global_logger as logger
import os
import pandas as pd
import numpy as np

from models.TabPFN import run_TabPFN_pipeline

'''
TODO:


1. put all ctr23 datasets, and put a table FOR ALL DATASETS train size, test size, number of input features, number of output feautres, number of catorgical features, number missing feautres, percentage missing
2. look at kiran paper new and update
3. talk to kiran and see his implemtnation and compare with folding

4. implement tabpfn, xgboost, catboost

'''

if __name__ == '__main__':

    train_TabResFlow = False
    train_TabPFN = True

    if train_TabResFlow:
        from models.TabResFlow import run_TabResFlow_pipeline

        overall_summary_df = run_TabResFlow_pipeline(
        source_dataset="uci",
        test_single_datasets = "yacht", # can specify a single dataset to test ie "yacht", otherwise None
        kaggle_training = False,
        # base_model_save_path_template="trained_models/tabresflow_best_{dataset_key}_fold{fold_idx}.pth"
        )

    if train_TabPFN:
        overall_summary_df = run_TabPFN_pipeline(
        source_dataset = "openml_ctr23",
        models_train_types = ["tabpfn_regressor" , "autotabpfn_regressor"],
        test_single_datasets = None, #361622
    )

    #print(overall_summary_df)