import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
RANDOM_STATE = 42

# ensure results exists
#os.makedirs(RESULTS_DIR, exist_ok=True)

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
        "gas": {"name": "GAS Sensor", "domain": "Chemistry"},
        #"hepmass": {"name": "HEPMASS", "domain": "Particle Physics"},
        #"miniboone": {"name": "MINIBOONE", "domain": "Neutrino Physics"},
        "power_grid": {"name": "POWER Grid", "domain": "Energy Systems"},
        "naval_multi": {"name": "Condition Based Maintenance of Naval Propulsion Plants", "domain": "Mechanical Engineering"},
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
        #"361268": {"name": "fps_benchmark"}, # this dataset really is bad, one hoy encoding it gives 958 columns!
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