import os
import random
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
# ensure results exists
#os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE) # for all GPUs


# one thing to note here that all datasets here are numerical, with the excpetion of openml_Ctr23
DATASETS = {
    # UCI datasets (From NodeFlow) (8)
    # raw datasets are already in the repo
    # (NodeFlow and kiran paper) both has target scaling of [-1,1] for all input features and target
    "uci": {
        "concrete": {"name": "Concrete Compressive Strength"},
        "energy": {"name": "Energy Efficiency"},
        "kin8nm": {"name": "Kinematics 8nm"},  # From OpenML, but grouped under UCI
        "naval-propulsion-plant": {"name": "Condition Based Maintenance of Naval Propulsion Plants"},
        "power-plant": {"name": "Combined Cycle Power Plant"},
        "protein-tertiary-structure": {"name": "Physicochemical Properties of Protein Tertiary Structure"},
        "wine-quality-red": {"name": "Wine Quality"},
        "yacht": {"name": "Yacht Hydrodynamics"},
    },
    # Multivariate datasets from Probabilistic Flow Circuits - UCI (4)
    # importnat note: the datasets can be found in the refrenced paper: 
    # Masked Autoregressive Flow for Density Estimation, https://arxiv.org/abs/1705.07057
    # the repo for it: https://github.com/gpapamak/maf/tree/master?tab=readme-ov-file#how-to-get-the-datasets, 
    # where datasets can be found and downlaoded, and so the pre-processing , i provided a method for download
    "multivariate": {
        "gas": {"name": "GAS Sensor"},
        #"hepmass": {"name": "HEPMASS"},
        #"miniboone": {"name": "MINIBOONE"},
        "power_grid": {"name": "POWER Grid"},
        "naval_multi": {"name": "Condition Based Maintenance of Naval Propulsion Plants"},
    },
    # OpenML-CTR23 datasets https://www.openml.org/search?type=study&study_type=undefined&sort=tasks_included&id=353
    "openml_ctr23": {
        "361251": {"name": "grid_stability"},
        "361252": {"name": "video_transcoding"},
        "361253": {"name": "wave_energy"},
        "361254": {"name": "sarcos"},
        "361255": {"name": "california_housing"},
        "361256": {"name": "cpu_activity"},
        "361257": {"name": "diamonds"},
        "361258": {"name": "kin8nm"},
        "361259": {"name": "pumadyn32nh"},
        "361260": {"name": "miami_housing"},
        "361261": {"name": "cps88wages"},
        "361264": {"name": "socmob"},
        "361266": {"name": "kings_county"},
        "361267": {"name": "brazilian_houses"},
        "361268": {"name": "fps_benchmark"},
        "361269": {"name": "health_insurance"},
        "361272": {"name": "fifa"},
        "361234": {"name": "abalone"},
        "361235": {"name": "airfoil_self_noise"},
        "361236": {"name": "auction_verification"},
        "361237": {"name": "concrete_compressive_strength"},
        "361241": {"name": "physiochemical_protein"},
        "361242": {"name": "superconductivity"},
        "361243": {"name": "geographical_origin_of_music"},
        "361244": {"name": "solar_flare"},
        "361247": {"name": "naval_propulsion_plant"},
        "361249": {"name": "white_wine"},
        "361250": {"name": "red_wine"},
        "361616": {"name": "Moneyball"},
        "361617": {"name": "energy_efficiency"},
        "361618": {"name": "forest_fires"},
        "361619": {"name": "student_performance_por"},
        "361621": {"name": "QSAR_fish_toxicity"},
        "361622": {"name": "cars"},
        "361623": {"name": "space_ga"}
    },
}

DATASET_MODEL_CONFIGS = {
    "concrete": {
        "TabResFlow_MODEL_HYPERPARAMS": {
            'embedding_dim_per_feature': 64, 'numerical_encoder_intermediate_dim': 100,
            'resnet_main_processing_dim': 128, 'resnet_depth': 6, 'resnet_block_hidden_factor': 1.0,
            'resnet_activation_dropout': 0.2, 'resnet_residual_dropout': 0.2,
            'flow_transforms': 5, 'flow_mlp_layers_in_transform': 2, 'flow_bins': 8,
            'categorical_cardinalities': None,
        },
        "TabResFlow_TRAIN_HYPERPARAMS": {
            'lr': 0.003, 'weight_decay': 1e-4, 'num_epochs': 400,
            'patience_early_stopping': 400, 'batch_size': 2048
        }
    },
    "energy": {
        "TabResFlow_MODEL_HYPERPARAMS": {
            'embedding_dim_per_feature': 64, 'numerical_encoder_intermediate_dim': 100,
            'resnet_main_processing_dim': 128, 'resnet_depth': 6, 'resnet_block_hidden_factor': 1.0,
            'resnet_activation_dropout': 0.2, 'resnet_residual_dropout': 0.2,
            'flow_transforms': 5, 'flow_mlp_layers_in_transform': 2, 'flow_bins': 8,
            'categorical_cardinalities': None,
        },
        "TabResFlow_TRAIN_HYPERPARAMS": {
            'lr': 0.003, 'weight_decay': 1e-4, 'num_epochs': 400,
            'patience_early_stopping': 400, 'batch_size': 2024 # As per your original script
        }
    },
    "kin8nm": {
        "TabResFlow_MODEL_HYPERPARAMS": {
            'embedding_dim_per_feature': 64, 'numerical_encoder_intermediate_dim': 100,
            'resnet_main_processing_dim': 256, 'resnet_depth': 6, 'resnet_block_hidden_factor': 1.0,
            'resnet_activation_dropout': 0.2, 'resnet_residual_dropout': 0.2,
            'flow_transforms': 5, 'flow_mlp_layers_in_transform': 2, 'flow_bins': 8,
            'categorical_cardinalities': None,
        },
        "TabResFlow_TRAIN_HYPERPARAMS": {
            'lr': 0.003, 'weight_decay': 1e-4, 'num_epochs': 400, # 60 in nodeflow?
            'patience_early_stopping': 400, 'batch_size': 2048
        }
    },
    "naval-propulsion-plant": {
        "TabResFlow_MODEL_HYPERPARAMS": {
            'embedding_dim_per_feature': 64, 'numerical_encoder_intermediate_dim': 100,
            'resnet_main_processing_dim': 128, 'resnet_depth': 6, 'resnet_block_hidden_factor': 1.0,
            'resnet_activation_dropout': 0.2, 'resnet_residual_dropout': 0.2,
            'flow_transforms': 5, 'flow_mlp_layers_in_transform': 2, 'flow_bins': 8,
            'categorical_cardinalities': None,
        },
        "TabResFlow_TRAIN_HYPERPARAMS": {
            'lr': 0.003, 'weight_decay': 1e-4, 'num_epochs': 300,
            'patience_early_stopping': 400, 'batch_size': 2048
        }
    },
    "power-plant": {
        "TabResFlow_MODEL_HYPERPARAMS": {
            'embedding_dim_per_feature': 64, 'numerical_encoder_intermediate_dim': 100,
            'resnet_main_processing_dim': 128, 'resnet_depth': 4, 'resnet_block_hidden_factor': 1.0,
            'resnet_activation_dropout': 0.2, 'resnet_residual_dropout': 0.2,
            'flow_transforms': 4, 'flow_mlp_layers_in_transform': 2, 'flow_bins': 8,
            'categorical_cardinalities': None,
        },
        "TabResFlow_TRAIN_HYPERPARAMS": {
            'lr': 0.003, 'weight_decay': 1e-4, 'num_epochs': 200,
            'patience_early_stopping': 400, 'batch_size': 2048
        }
    },
    "protein-tertiary-structure": {
        "TabResFlow_MODEL_HYPERPARAMS": {
            'embedding_dim_per_feature': 64, 'numerical_encoder_intermediate_dim': 100,
            'resnet_main_processing_dim': 128, 'resnet_depth': 8, 'resnet_block_hidden_factor': 1.0,
            'resnet_activation_dropout': 0.3, 'resnet_residual_dropout': 0.3,
            'flow_transforms': 5, 'flow_mlp_layers_in_transform': 2, 'flow_bins': 10,
            'categorical_cardinalities': None,
        },
        "TabResFlow_TRAIN_HYPERPARAMS": {
            'lr': 0.003, 'weight_decay': 1e-4, 'num_epochs': 100,
            'patience_early_stopping': 400, 'batch_size': 2048
        }
    },
    "wine-quality-red": {
        "TabResFlow_MODEL_HYPERPARAMS": {
            'embedding_dim_per_feature': 93, 'numerical_encoder_intermediate_dim': 100,
            'resnet_main_processing_dim': 51, 'resnet_depth': 3, 'resnet_block_hidden_factor': 1.0,
            'resnet_activation_dropout': 0.2, 'resnet_residual_dropout': 0.49,
            'flow_transforms': 13, 'flow_mlp_layers_in_transform': 2, 'flow_bins': 8,
            'categorical_cardinalities': None,
        },
        "TabResFlow_TRAIN_HYPERPARAMS": {
            'lr': 0.003, 'weight_decay': 1e-4, 'num_epochs': 400,
            'patience_early_stopping': 400, 'batch_size': 2048
        }
    },
    "yacht": {
        "TabResFlow_MODEL_HYPERPARAMS": {
            'embedding_dim_per_feature': 64, 'numerical_encoder_intermediate_dim': 100,
            'resnet_main_processing_dim': 128, 'resnet_depth': 3, 'resnet_block_hidden_factor': 1.0,
            'resnet_activation_dropout': 0.3, 'resnet_residual_dropout': 0.3,
            'flow_transforms': 13, 'flow_mlp_layers_in_transform': 2, 'flow_bins': 8,
            'categorical_cardinalities': None,
        },
        "TabResFlow_TRAIN_HYPERPARAMS": {
            'lr': 0.003, 'weight_decay': 1e-4, 'num_epochs': 400,
            'patience_early_stopping': 400, 'batch_size': 1024
        }
    },
    "openML-general": {
        "TabResFlow_MODEL_HYPERPARAMS": {
            'embedding_dim_per_feature': 128, 'numerical_encoder_intermediate_dim': 100,
            'resnet_main_processing_dim': 128, 'resnet_depth': 6, 'resnet_block_hidden_factor': 1.0,
            'resnet_activation_dropout': 0.2, 'resnet_residual_dropout': 0.2,
            'flow_transforms': 10, 'flow_mlp_layers_in_transform': 2, 'flow_bins': 8,
            'categorical_cardinalities': None,
        },
        "TabResFlow_TRAIN_HYPERPARAMS": {
            'lr': 0.03, 'weight_decay': 1e-4, 'num_epochs': 400,
            'patience_early_stopping': 100, 'batch_size': 1024
        }
    },
}