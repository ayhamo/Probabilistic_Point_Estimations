import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
RANDOM_STATE = 42

# ensure results exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# one thing to note here that all datasets here are numerical, with the excpetion of openml_Ctr23
DATASETS = {
    # UCI datasets (From NodeFlow) (8)
    # datasets can be downloaded using the code, pre-prcoessing were taken from NodeFlow paper
    # (NodeFlow and kiran paper) both has target scaling of [-1,1] for all input features and target
    "uci": {
        "concrete": {"name": "Concrete Compressive Strength", "domain": "Civil Engineering"},
        "energy": {"name": "Energy Efficiency", "domain": "Building Science"},
        "naval": {"name": "Condition Based Maintenance of Naval Propulsion Plants", "domain": "Mechanical Engineering"},
        "power": {"name": "Combined Cycle Power Plant", "domain": "Energy"},
        "protein": {"name": "Physicochemical Properties of Protein Tertiary Structure", "domain": "Biophysics"},
        "wine": {"name": "Wine Quality", "domain": "Food Science"},
        "yacht": {"name": "Yacht Hydrodynamics", "domain": "Fluid Dynamics"},
        "kin8nm": {"name": "Kinematics 8nm", "domain": "Robotics"}  # From OpenML, but grouped under UCI
    },
    # Multivariate datasets from Probabilistic Flow Circuits - UCI (4)
    # importnat note: the datasets can be found in the refrenced paper: 
    # Masked Autoregressive Flow for Density Estimation, https://arxiv.org/abs/1705.07057
    # the repo for it: https://github.com/gpapamak/maf/tree/master?tab=readme-ov-file#how-to-get-the-datasets, 
    # where datasets can be found and downlaoded, and so the pre-processing , i provided a method for download
    "multivariate": {
        "gas": {"name": "GAS Sensor", "domain": "Chemistry"},
        #"hepmass": {"name": "HEPMASS", "domain": "Particle Physics"},
        #"miniboone": {"name": "MINIBOONE", "domain": "Neutrino Physics"},
        "power_grid": {"name": "POWER Grid", "domain": "Energy Systems"},
        "naval_multi": {"name": "Condition Based Maintenance of Naval Propulsion Plants", "domain": "Mechanical Engineering"},
    },
    # OpenML-CTR23 datasets (9) https://www.openml.org/search?type=study&study_type=undefined&sort=tasks_included&id=353
    "openml_ctr23": {
        "44958": {"name": "auction_verification", "domain": "Auctions/Economics"},
        "44957": {"name": "airfoil_self_noise", "domain": "Engineering"},
        "44990": {"name": "brazilian_houses", "domain": "Real Estate"},
        "44994": {"name": "cars", "domain": "Automotive"},
        "44962": {"name": "forest_fires", "domain": "Environmental"}, # this has month and day, but paper says explicitly to one hot encode them
        "44970": {"name": "QSAR_fish_toxicity", "domain": "Chemistry/Toxicology"},
        "45402": {"name": "space_ga", "domain": "Astronomy/Physics"},
        "44987": {"name": "socmob", "domain": "Social Science"},
        "44965": {"name": "geographical_origin_of_music", "domain": "Music/ML Benchmark"}  # High-dimensional, small-n datasets
    },
}


POINT_ESTIMATOR_MODELS = {
    "LinearRegression": {}
    # "RandomForestRegressor": {"n_estimators": 100, "random_state": 42}
}

PROBABILISTIC_MODELS = {
    # "GaussianProcessRegressor": {}
}

# Metrics to compute
METRICS = ['MAE', 'MSE', 'MAPE'] # For point estimates
PROBABILISTIC_METRICS = ['NLL', 'CRPS', 'EnergyScore'] # For probabilistic forecasts