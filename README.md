# Will be filled at later date, Currently will host results

The runs are found in https://www.kaggle.com/code/ayhamo/thesis-main in Version tab

For My datasets: 
- UCI ones all have 20 folds for each dataset, but protien has 5 folds.
- OpenML-CTR23 all have 10 folds.

# Questions and answers


### 1. How does MSE(ŷP) compare to MSE(f(x)) in practice? Is the assumption MSE(ŷP) ≥ MSE(f(x)) generally true empirically?

Yes, the assumption that MSE(ŷP) ≥ MSE(f(x)) is generally true empirically.

The SOTA point predictors, specifically TabPFN and XGBoost, consistently achieved lower MSE and RMSE values than the point estimates derived from any of the probabilistic models across the majority of both UCI and OpenML-CTR23 datasets. This confirms that models directly optimized for point-wise error are, on average, more accurate for the specific current task.

However, the key finding is that this is not a universal saying (which is why i asked genrally!) but a strong trend. the performance gap varies significantly, and in several cases, a top-tier probabilistic model like Thin and Deep Gaussian Processes (TDGPs) produced point estimates that were highly competitive, with negligible margin or even outperforming some of the point predictors (color me suprised, since that paper is recent[NeurIPS 2023] and the one i stuggeld to run the most).

### 2. How do different classes of probabilistic models perform in terms of derived point estimates?

There is a clear and big performance hierarchy among the different classes of probabilistic models:

*   **Top (Competitive): Thin and Deep Gaussian Processes (TDGPs)** was the strongest performer. the derived point estimates were consistently the best among the probabilistic models and were often competitive with the dedicated point predictors like CatBoost and XGBoost.
*   **Reliable (Good Baseline): TabResFlow** solid and reliable model. Its point estimates were generally reasonable and better than the more problematic models, but they did not typically challenge the top-tier point predictors.
*   **Problematic (Unreliable):**
    *   **Tabular Transformer VAE (TTVAE):** This model was difficult to evaluate fairly. While its NLL metric was invalid (due to optimizing the ELBO), its CRPS scores were often respectable. However, its derived point estimates were not competitive with the top models.
    *   **Auto-Regressive Moving Diffusion Models (ARMD):** This model class failed badly for this task. The MSE, RMSE, and MAE values were orders of magnitude worse than all other models, rendering its point estimates completely unusable. 

    This is a critical finding that demonstrates that not all advanced generative architectures are suitable for general-purpose tabular regression.

### 3. Does good performance on probabilistic metrics (NLL, CRPS) correlate with good performance on point metrics (MSE, MAE)?

Yes, there is a moderate positive correlation, but it is not perfect and comes with some side notes.

*   **CRPS is a more reliable indicator than NLL.** Generally, models that achieved a low (good) CRPS also had a low (good) MSE. This makes sense, as CRPS evaluates the entire predictive distribution's accuracy and calibration. It is difficult for a model to have a good CRPS if the center of its predicted distribution (i.e., its point estimate) is far from the true value.
*   **NLL proved to be a problematic correlator** For TTVAE, the NLL was an invalid metric. For ARMD, the NLL was extremely poor, which did correlate with its terrible MSE. For a well-behaved model like TabResFlow, good NLL generally corresponded to good MSE.

### 4. Investigate the correlation and trade-offs between performance on probabilistic metrics and point estimation metrics. Under which conditions are probabilistic models better?

The models that exclusively minimized RMSE (XGBoost, TabPFN) generally had the best RMSE.

Point estimates derived from probabilistic models were likely to be better than dedicated point predictors under two  conditions:

1.  **When the Probabilistic Model is Powerful:** The only probabilistic model that consistently challenged the point predictors was TDGPs. This suggests that only well made probabilistic architectures can overcome the disadvantage of not directly optimizing for a point-wise loss function.
2.  **On Datasets with Potentially Complex Structures:** While not definitively proven, it is possible that datasets like `kin8nm`, where TDGPs performed very well, may have characteristics that are better captured by a flexible model like a Gaussian Process. A GP can learn a more accurate representation of the underlying function, leading to a more accurate mean prediction, whereas models that implicitly assume a simpler error distribution might be disadvantaged.

### 5. Relationship (or lack of) and the variance between probabilistic performance and point estimation accuracy in practice.

The relationship can be described as a "soft hierarchy". Better probabilistic performance (especially low CRPS) generally leads to better point estimation accuracy. However, there is a "specialist's edge," where models that focus solely on point prediction can often fine-tune their results to be slightly better on that specific metric.

one imporant thing to note is that the variance in performance is extremely high, particularly across the different architectures of probabilistic models.
*   The gap between a top-performing probabilistic model (TDGPs) and a failing one (ARMD) is huge, basiclly that the choice of architecture is far more critical than in the more mature field of point predictors.
*   The variance in results across the 35 OpenML-CTR23 datasets for all models indicates that tabular data in the wild is incredibly diverse and challenging, with no single model winning universally.

### 6. Identify conditions (datasets, model types) where probabilistic models offer competitive or superior point estimates.

*   **Model Type:** The only model type that consistently offered competitive or superior point estimates was Thin and Deep Gaussian Processes (TDGPs) (but i also have to mention that it consomes the most resoruces and hardest to setup!).
*   **Datasets (Superior):** The most notable one is `kin8nm`, where TDGPs achieved an RMSE (0.0636) that was better than CatBoost (0.1042) and competitive with TabPFN (0.0704).
*   **Datasets (Competitive):** On many of the classic UCI datasets like `Concrete`, `Energy`, and `Power`, the performance gap between TDGPs and the point predictors was very small. In these cases, the minor (if any) sacrifice in point accuracy would be well worth the significant benefit of gaining a full probabilistic forecast.

### 7. Small Table Ranking All the Methods

This table ranks the models based on their overall performance and reliability for their intended purpose, as demonstrated in your results.

| **Tier** | **Model(s)** | **Primary Role** | **Strengths & Weaknesses** |
| :--- | :--- | :--- | :--- |
| **Tier 1: SOTA Point Prediction** | TabPFN, XGBoost | Point Estimation | **Strengths:** Consistently highest point accuracy (lowest RMSE/MAE). Robust and reliable. <br> **Weaknesses:** Provide no direct uncertainty information. |
| **Tier 2: Competitive All-Rounders** | TDGPs, CatBoost, TabResFlow | Probabilistic & Point | **Strengths:** TDGPs offer excellent probabilistic forecasts with highly competitive point accuracy. CatBoost is a robust point predictor. TabResFlow is a reliable probabilistic baseline. <br> **Weaknesses:** TDGPs can be difficult to implement. |
| **Tier 3: Good but Outperformed** | TabResNet | Point Estimation | **Strengths:** A decent deep learning baseline. <br> **Weaknesses:** Generally outperformed by gradient boosting and TabPFN on point accuracy. |
| **Tier 4: Niche / Problematic** | TTVAE | Probabilistic | **Strengths:** Can produce reasonable probabilistic forecasts (based on CRPS). <br> **Weaknesses:** Unreliable NLL metric makes evaluation difficult; not competitive on point accuracy. |
| **Tier 5: Failed in this Context** | ARMD | Probabilistic | **Strengths:** None observed in this study. <br> **Weaknesses:** Catastrophic failure across all metrics, indicating it is unsuitable for this type of task. |

# Per Metric Final results


### Test NLL

will be updated with new ttvae results

### Test CRPS
| Dataset | TabResFlow | Tabular Transformer Variational Autoencoder (TTVAE) | Thin and Deep Gaussian Processes (TDGPs) | Auto-Regressive Moving Diffusion Models (ARMD) | TabPFN | XGBoost | CatBoost | TabResNet |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| Concrete | 2.4417 ± 0.3120 | 16.5084 ± 4.2591 | 2.4322 ± 0.1734 | 28941764.7000 ± 645085.9229 | nan | nan | nan | nan |
| Energy | 0.2943 ± 0.0322 | 10.0961 ± 1.0014 | 0.3639 ± 0.0155 | 8692267.9750 ± 349840.2804 | nan | nan | nan | nan |
| Kin8nm | 0.0387 ± 0.0013 | 0.0360 ± 0.0059 | 0.0372 ± 0.0008 | 4844.7940 ± 51.9951 | nan | nan | nan | nan |
| Moneyball (361616) | 39.3231 ± 31.8497 | 81.2682 ± 7.6853 | 67.6429 ± 0.5373 | 1502193.3875 ± 34019.5532 | nan | nan | nan | nan |
| Naval | 0.0007 ± 0.0001 | 0.0004 ± 0.0001 | 0.0043 ± 0.0012 | 189685965619.2000 ± 2510031123.3265 | nan | nan | nan | nan |
| Power | 1.9334 ± 0.0505 | 0.5379 ± 0.1704 | 2.1911 ± 0.0664 | 1005387.5375 ± 11128.2658 | nan | nan | nan | nan |
| Protein | 1.6971 ± 0.0168 | 0.4028 ± 0.4725 | 2.4666 ± 0.0265 | 206581632126156.8125 ± 1679975204008.8855 | nan | nan | nan | nan |
| QSAR_fish_toxicity (361621) | 0.4836 ± 0.0403 | 1.1546 ± 0.1580 | 0.5936 ± 0.0266 | 5886.5473 ± 93.6446 | nan | nan | nan | nan |
| Wine | 0.3172 ± 0.0216 | 0.6452 ± 0.0955 | 0.4034 ± 0.0416 | 657373.1438 ± 27750.8696 | nan | nan | nan | nan |
| Yacht | 0.5382 ± 0.2553 | 9.7021 ± 3.2082 | 0.4563 ± 0.0278 | 216154.6469 ± 21596.0144 | nan | nan | nan | nan |
| abalone (361234) | 1.0597 ± 0.0524 | 0.4786 ± 0.0803 | 1.3956 ± 0.0625 | 6852.1713 ± 177.3544 | nan | nan | nan | nan |
| airfoil_self_noise (361235) | 1.7791 ± 0.3934 | 6.6162 ± 0.7420 | 0.7271 ± 0.0659 | 11163358617.6000 ± 365914748.0158 | nan | nan | nan | nan |
| auction_verification (361236) | 1864.6033 ± 1120.3322 | 6816.8794 ± 712.0473 | 484.1481 ± 30.6753 | 41369319424.0000 ± 1017404228.8739 | nan | nan | nan | nan |
| brazilian_houses (361267) | 273592.3841 ± 475166.7514 | 3728.7975 ± 452.2699 | 3360.5704 ± 836.8226 | 109.2335 ± 6.6664 | nan | nan | nan | nan |
| california_housing (361255) | 25868.2331 ± 1542.1326 | 2192.9962 ± 507.7850 | 24902.5952 ± 618.9691 | 11745.2175 ± 115.2010 | nan | nan | nan | nan |
| cars (361622) | 1704.8845 ± 935.2053 | 7887.3716 ± 1166.7600 | 5842.3305 ± 2434.2272 | 59242753228.8000 ± 2997495578.6556 | nan | nan | nan | nan |
| concrete_compressive_strength (361237) | 3.0754 ± 0.4363 | 15.3844 ± 3.7464 | 2.3489 ± 0.3038 | 28879276.8000 ± 666461.1526 | nan | nan | nan | nan |
| cpu_activity (361256) | 4.2195 ± 4.5878 | 2.3023 ± 0.3296 | 1.2421 ± 0.0610 | 30161.7643 ± 216.4013 | nan | nan | nan | nan |
| cps88wages (361261) | 1771.2695 ± 1798.8748 | 47.4844 ± 102.5393 | 201.8928 ± 2.8695 | 27.9386 ± 0.1550 | nan | nan | nan | nan |
| diamonds (361257) | 3516.9083 ± 5325.3385 | 332.8236 ± 138.7593 | 2785.3115 ± 19.9483 | 122.9909 ± 0.7521 | nan | nan | nan | nan |
| energy_efficiency (361617) | 1.1639 ± 0.4974 | 9.9136 ± 0.7780 | 0.3562 ± 0.0186 | 8684711.6500 ± 257250.0924 | nan | nan | nan | nan |
| fifa (361272) | 90946.3945 ± 78775.3609 | 8152.2665 ± 414.4364 | 60.3657 ± 0.0000 | 0± 0 | nan | nan | nan | nan |
| forest_fires (361618) | 69.4872 ± 64.3380 | 12.9297 ± 7.4028 | 19.0932 ± 9.0204 | 15487436.6000 ± 862946.9561 | nan | nan | nan | nan |
| fps_benchmark (361268) | 29.9009 ± 57.5754 | 52.1407 ± 3.5818 | 39.5227 ± 0.5548 | 20822.2123 ± 156.5238 | nan | nan | nan | nan |
| geographical_origin_of_music (361243) | 13.2288 ± 13.1600 | 15.5612 ± 1.3148 | 13.8535 ± 0.6738 | 25733.6937 ± 835.3423 | nan | nan | nan | nan |
| grid_stability (361251) | 0.0065 ± 0.0017 | 0.0055 ± 0.0008 | 0.0024 ± 0.0002 | 1.0433 ± 0.0050 | nan | nan | nan | nan |
| health_insurance (361269) | 9.0640 ± 0.9230 | 4.2734 ± 0.8808 | 15.9719 ± 0.1980 | 2.2112 ± 0.0118 | nan | nan | nan | nan |
| kin8nm (361258) | 0.0445 ± 0.0044 | 0.0343 ± 0.0067 | 0.0359 ± 0.0010 | 0.7985 ± 0.0048 | nan | nan | nan | nan |
| kings_county (361266) | 2566143.5319 ± 3258756.7758 | 261336.6678 ± 30781.0411 | 214278.6648 ± 4125.2262 | 2196.9810 ± 15.8375 | nan | nan | nan | nan |
| miami_housing (361260) | 226725.1161 ± 240388.6019 | 51152.5223 ± 7612.4019 | 35180.3496 ± 527.1171 | 20099.2022 ± 111.6660 | nan | nan | nan | nan |
| naval_propulsion_plant (361247) | 0.0017 ± 0.0005 | 0.0004 ± 0.0001 | 0.0052 ± 0.0018 | 216084057292.8000 ± 3938915264.8445 | nan | nan | nan | nan |
| physiochemical_protein (361241) | 2.1946 ± 0.1644 | 1.1057 ± 2.2849 | 2.4803 ± 0.0254 | 208335671604019.1875 ± 2586883697441.3501 | nan | nan | nan | nan |
| pumadyn32nh (361259) | 0.0127 ± 0.0004 | 0.0269 ± 0.0007 | 0.0271 ± 0.0004 | 7.0070 ± 0.0373 | nan | nan | nan | nan |
| red_wine (361250) | 0.3855 ± 0.0194 | 0.6226 ± 0.1050 | 0.4082 ± 0.0314 | 647448.3562 ± 29165.2181 | nan | nan | nan | nan |
| sarcos (361254) | 15.5040 ± 20.0312 | 5.0657 ± 0.5347 | 1.6973 ± 0.0302 | 3.4685 ± 0.0209 | nan | nan | nan | nan |
| socmob (361264) | 23.4827 ± 21.4010 | 17.4444 ± 3.6711 | 18.2049 ± 1.5463 | 1.2579 ± 0.0421 | nan | nan | nan | nan |
| solar_flare (361244) | 0.6500 ± 0.3381 | 0.3005 ± 0.0633 | 0.4636 ± 0.0595 | 953.5112 ± 13.5871 | nan | nan | nan | nan |
| space_ga (361623) | 0.0642 ± 0.0076 | 0.0309 ± 0.0079 | 0.0581 ± 0.0048 | 145129825995902144.0000 ± 2368378464024677.0000 | nan | nan | nan | nan |
| student_performance_por (361619) | 2.3243 ± 2.3020 | 2.7032 ± 0.3712 | 2.3706 ± 0.3717 | 6097.2813 ± 216.5593 | nan | nan | nan | nan |
| superconductivity (361242) | 11.3786 ± 14.5353 | 9.0065 ± 0.3982 | 27.4435 ± 0.5922 | 6067316633.6000 ± 77265609.9660 | nan | nan | nan | nan |
| video_transcoding (361252) | 16.6803 ± 20.9636 | 1.1742 ± 0.9249 | 8.5438 ± 0.1180 | 2508689.0727 ± 18507.5873 | nan | nan | nan | nan |
| wave_energy (361253) | 231868.0315 ± 463776.9219 | 45547.0007 ± 5340.1363 | 82569.0760 ± 913.4778 | nan | nan | nan | nan | nan |
| white_wine (361249) | 0.3997 ± 0.0170 | 0.1406 ± 0.0193 | 0.4328 ± 0.0237 | 1159133.4000 ± 20821.1927 | nan | nan | nan | nan |

### Test MSE
| Dataset | TabResFlow | Tabular Transformer Variational Autoencoder (TTVAE) | Thin and Deep Gaussian Processes (TDGPs) | Auto-Regressive Moving Diffusion Models (ARMD) | TabPFN | XGBoost | CatBoost | TabResNet |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| Concrete | 27.4820 ± 5.8639 | nan | 21.7218 ± 4.8990 | 5788.3535 ± 129.0174 | 15.2815 ± 4.5155 | 19.3543 ± 5.1196 | 21.9038 ± 4.6034 | 33.4399 ± 5.1269 |
| Energy | 0.4434 ± 0.2697 | nan | 0.2915 ± 0.0588 | 1738.4543 ± 69.9680 | 0.1829 ± 0.0555 | 0.1085 ± 0.0490 | 0.2511 ± 0.0670 | 33.0509 ± 53.7260 |
| Kin8nm | 0.0053 ± 0.0004 | nan | 0.0041 ± 0.0002 | 0.9697 ± 0.0104 | 0.0050 ± 0.0002 | 0.0151 ± 0.0005 | 0.0109 ± 0.0006 | 0.0098 ± 0.0026 |
| Moneyball (361616) | 31827.1191 ± 82224.6953 | nan | 8206.9628 ± 169.9284 | 300.4394 ± 6.8039 | 437.3483 ± 31.5674 | 556.7851 ± 42.9726 | 945.3987 ± 78.6804 | 7313.9748 ± 1285.6976 |
| Naval | 0.0000 ± 0.0000 | nan | 0.0001 ± 0.0000 | 37937192.0000 ± 502006.6562 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0002 ± 0.0002 |
| Power | 16.2194 ± 1.4487 | nan | 14.6059 ± 1.4419 | 201.0782 ± 2.2256 | 9.0807 ± 1.4087 | 9.4882 ± 1.4220 | 13.3446 ± 1.2729 | 106.1714 ± 90.4414 |
| Protein | 19.0796 ± 0.5625 | nan | 15.4953 ± 0.3015 | 41316327424.0000 ± 335995072.0000 | 12.4398 ± 0.1591 | 11.2619 ± 0.0928 | 17.2313 ± 0.2199 | 58.5130 ± 28.5243 |
| QSAR_fish_toxicity (361621) | 0.9350 ± 0.0616 | nan | 1.0263 ± 0.1517 | 1.1780 ± 0.0187 | 0.7368 ± 0.0981 | 0.7874 ± 0.1119 | 0.7991 ± 0.1163 | 0.9016 ± 0.1593 |
| Wine | 0.4846 ± 0.0479 | nan | 0.4812 ± 0.0949 | 131.4754 ± 5.5502 | 0.3927 ± 0.0421 | 0.3205 ± 0.0552 | 0.3617 ± 0.0518 | 0.4706 ± 0.0462 |
| Yacht | 2.7534 ± 3.8523 | nan | 0.3250 ± 0.1653 | 43.2317 ± 4.3192 | 0.2015 ± 0.1627 | 0.8450 ± 0.8946 | 2.6731 ± 4.7011 | 12.0495 ± 8.3310 |
| abalone (361234) | 5.4463 ± 0.6886 | nan | 5.8646 ± 0.5809 | 1.3712 ± 0.0355 | 4.2753 ± 0.5141 | 4.9362 ± 0.4456 | 5.6123 ± 0.5619 | 9.6863 ± 2.7162 |
| airfoil_self_noise (361235) | 21.6791 ± 30.4767 | nan | 1.8531 ± 0.3224 | 2232671.7500 ± 73182.9297 | 0.9484 ± 0.2265 | 1.6446 ± 0.4137 | 3.8381 ± 0.7613 | 40.4833 ± 8.2148 |
| auction_verification (361236) | 37397924 ± 35045820 | nan | 819562.0248 ± 169028.9401 | 8273864.0000 ± 203480.6875 | 351583.2517 ± 86421.6278 | 256318.9940 ± 100614.9921 | 1273506.0958 ± 264753.6442 | 710135.0436 ± 140923.8559 |
| brazilian_houses (361267) | 566601252864 ± 980091994112 | nan | 453149611.6628 ± 548600000.0000 | 12039475.0000 ± 5197279.5000 | 118880180.7745 ± 206370607.5488 | 27117261.1990 ± 28250139.7710 | 126339454.3863 ± 187933986.3218 | 184088113.1992 ± 360229625.4841 |
| california_housing (361255) | 3053801728 ± 551256192 | nan | 2114569869.4119 ± 176134436.5426 | 1948064768.0000 ± 38024968.0000 | 1847555008.9483 ± 191653405.8820 | 1987546270.0629 ± 174126527.4015 | 2343352218.6663 ± 181510180.0063 | 5731829172.8496 ± 2219063887.9075 |
| cars (361622) | 27622500 ± 21774774 | nan | 69166185.4658 ± 35316919.5792 | 11848550.0000 ± 599499.0625 | 4209088.5973 ± 649921.0897 | 5747673.8650 ± 1236742.8127 | 4543009.6012 ± 776057.9047 | 604268570.6450 ± 748883264.5074 |
| concrete_compressive_strength (361237) | 74.3113 ± 128.0850 | nan | 22.8821 ± 5.1974 | 5775.8564 ± 133.2921 | 14.9270 ± 6.7669 | 18.5999 ± 6.4485 | 21.3442 ± 6.7796 | 32.1677 ± 6.8079 |
| cpu_activity (361256) | 166.3508 ± 104.4023 | nan | 5.5134 ± 0.6063 | 15199808512.0000 ± 266567056.0000 | 7.1112 ± 0.6237 | 4.8418 ± 0.3768 | 7.2375 ± 2.3588 | 39.2216 ± 58.7860 |
| cps88wages (361261) | 169979936 ± 378400000 | nan | 166730.7228 ± 40075.5367 | 20200.8398 ± 753.1591 | 146638.2187 ± 34645.6972 | 174594.3211 ± 35460.1001 | 146472.7367 ± 34682.6379 | 162233.0056 ± 33916.8092 |
| diamonds (361257) | 37594632 ± 46169312 | nan | 15782666.9924 ± 337309.1582 | 773309.8750 ± 8618.9453 | 281123.3248 ± 19605.7276 | 295368.5227 ± 24912.5720 | 815862.1613 ± 32506.8275 | 520249.2515 ± 56704.3436 |
| energy_efficiency (361617) | 5.5001 ± 4.4427 | nan | 0.2662 ± 0.0643 | 1736.9431 ± 51.4500 | 0.1692 ± 0.0479 | 0.0873 ± 0.0430 | 0.2313 ± 0.0661 | 13.6008 ± 9.9648 |
| fifa (361272) | 107799552000 ± 199156842496 | nan | 350067624.6939 ± 0.0000 | 0± 0 | 108425961.4904 ± 23289032.7671 | 83761065.2055 ± 14617882.5601 | 85819685.8628 ± 14559520.1413 | 152383782.9666 ± 24970103.3691 |
| forest_fires (361618) | 480273.5938 ± 1284821.1250 | nan | 8707.2220 ± 9348.0642 | 3097.4878 ± 172.5894 | 4063.1253 ± 7119.5515 | 4983.3606 ± 6490.6476 | 4224.7676 ± 6231.5007 | 5588.7016 ± 5658.2627 |
| fps_benchmark (361268) | 54526.1332 ± 122610.6687 | nan | 2958.6520 ± 88.0224 | 73166684160.0000 ± 1039000896.0000 | 3.1837 ± 0.6340 | 1.4339 ± 0.5376 | 9.9276 ± 1.3137 | 50.2346 ± 43.5918 |
| geographical_origin_of_music (361243) | 313.2901 ± 106.4647 | nan | 334.6108 ± 46.1712 | 5.1475 ± 0.1671 | 210.0773 ± 53.8257 | 242.9032 ± 51.3564 | 237.6736 ± 50.8530 | 301.4084 ± 62.1681 |
| grid_stability (361251) | 0.0002 ± 0.0001 | nan | 0.0000 ± 0.0000 | 3.2100 ± 0.0396 | 0.0000 ± 0.0000 | 0.0001 ± 0.0000 | 0.0001 ± 0.0000 | 0.0001 ± 0.0000 |
| health_insurance (361269) | 1020.0330 ± 1959.5533 | nan | 349.6880 ± 7.7430 | 52.5813 ± 0.7807 | 209.2058 ± 5.5444 | 276.2371 ± 9.3639 | 210.2858 ± 6.1212 | 331.6207 ± 10.4357 |
| kin8nm (361258) | 0.0072 ± 0.0016 | nan | 0.0038 ± 0.0002 | 0.9719 ± 0.0134 | 0.0049 ± 0.0003 | 0.0150 ± 0.0008 | 0.0105 ± 0.0005 | 0.0072 ± 0.0021 |
| kings_county (361266) | 14580080379298.2305 ± 21983366501781.0586 | nan | 130750469599.8554 ± 21531718468.4040 | 1339364608.0000 ± 41483080.0000 | 14132590527.7067 ± 4757652673.5136 | 14563350552.6073 ± 4486980316.7839 | 19785103189.0316 ± 5285839068.8266 | 33959113254.2392 ± 10880851510.4136 |
| miami_housing (361260) | 437507096576 ± 868867506176 | nan | 9090482510.0571 ± 1456990785.1374 | 8340740608.0000 ± 146972224.0000 | 7570139372.3785 ± 618035857.1107 | 7308453452.7606 ± 617317178.3352 | 7943239375.9150 ± 694338414.4499 | 12115593482.2717 ± 1902481405.5462 |
| naval_propulsion_plant (361247) | 0 ± 0 | nan | 0.0001 ± 0.0000 | 43216808.0000 ± 787784.8125 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0002 ± 0.0001 |
| physiochemical_protein (361241) | 24.4014 ± 1.6434 | nan | 15.9822 ± 0.1843 | 41667137536.0000 ± 517376928.0000 | 12.2795 ± 0.2773 | 11.1340 ± 0.2087 | 16.7753 ± 0.2781 | 52.2450 ± 30.8531 |
| pumadyn32nh (361259) | 0.0006 ± 0.0001 | nan | 0.0013 ± 0.0000 | 370.7990 ± 4.9482 | 0.0004 ± 0.0000 | 0.0005 ± 0.0000 | 0.0004 ± 0.0000 | 0.0009 ± 0.0003 |
| red_wine (361250) | 0.6350 ± 0.2270 | nan | 0.4668 ± 0.0494 | 129.4904 ± 5.8330 | 0.3901 ± 0.0467 | 0.3067 ± 0.0454 | 0.3542 ± 0.0412 | 0.4552 ± 0.0694 |
| sarcos (361254) | 889.9745 ± 769.7089 | nan | 10.6668 ± 0.2716 | 61.5187 ± 0.7155 | 7.7924 ± 0.9086 | 4.9011 ± 0.2683 | 13.8816 ± 0.7761 | 11.7916 ± 2.0053 |
| socmob (361264) | 13957.3438 ± 27060.8379 | nan | 1064.5726 ± 266.0666 | 117.1548 ± 9.6103 | 183.6488 ± 195.4986 | 240.1539 ± 206.5902 | 534.6711 ± 377.6231 | 983.5829 ± 600.8121 |
| solar_flare (361244) | 2.9819 ± 2.8886 | nan | 0.7965 ± 0.3175 | 0.1914 ± 0.0027 | 0.6290 ± 0.2475 | 0.8863 ± 0.1757 | 0.6533 ± 0.2302 | 0.8453 ± 0.2414 |
| space_ga (361623) | 0.0127 ± 0.0035 | nan | 0.0091 ± 0.0017 | 29025963606016.0000 ± 473676480512.0000 | 0.0088 ± 0.0041 | 0.0123 ± 0.0050 | 0.0127 ± 0.0046 | 0.0207 ± 0.0065 |
| student_performance_por (361619) | 11.6036 ± 9.4291 | nan | 11.6190 ± 4.0877 | 1.2202 ± 0.0433 | 7.4567 ± 2.7828 | 7.6262 ± 2.6984 | 7.2460 ± 2.2857 | 11.0542 ± 2.3991 |
| superconductivity (361242) | 1170.5513 ± 2171.2662 | nan | 1178.3707 ± 37.2792 | 1213463.3750 ± 15453.0752 | 94.9896 ± 6.7674 | 85.3380 ± 5.5108 | 118.3788 ± 6.1156 | 221.0088 ± 35.7952 |
| video_transcoding (361252) | 16063.2969 ± 19043.5284 | nan | 263.3198 ± 13.5337 | 293662751719424.0000 ± 4380295954432.0000 | 12.9445 ± 2.0629 | 0.7008 ± 0.2700 | 4.7962 ± 0.5461 | 2.5143 ± 0.7170 |
| wave_energy (361253) | 22847758336 ± 31917230080 | nan | 12621949056.0449 ± 175402380.8487 | nan | 20177032.2934 ± 4025032.8038 | 468047280.7898 ± 12394958.3820 | 76239808.9842 ± 1503636.3610 | 630536198.4836 ± 367691990.0705 |
| white_wine (361249) | 0.6213 ± 0.0533 | nan | 0.5054 ± 0.0524 | 231.8274 ± 4.1642 | 0.4841 ± 0.0471 | 0.3384 ± 0.0298 | 0.4443 ± 0.0291 | 0.5438 ± 0.0442 |

### Test RMSE
| Dataset | TabResFlow | Tabular Transformer Variational Autoencoder (TTVAE) | Thin and Deep Gaussian Processes (TDGPs) | Auto-Regressive Moving Diffusion Models (ARMD) | TabPFN | XGBoost | CatBoost | TabResNet |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| Concrete | 5.2117 ± 0.5660 | nan | 4.6327 ± 0.5094 | 76.0765 ± 0.8481 | 3.8640 ± 0.5923 | 4.3590 ± 0.5944 | 4.6536 ± 0.4982 | 5.7658 ± 0.4423 |
| Energy | 0.6453 ± 0.1645 | nan | 0.5371 ± 0.0553 | 41.6863 ± 0.8423 | 0.4229 ± 0.0636 | 0.3215 ± 0.0718 | 0.4969 ± 0.0651 | 4.7157 ± 3.2883 |
| Kin8nm | 0.0729 ± 0.0030 | nan | 0.0636 ± 0.0014 | 0.9847 ± 0.0053 | 0.0704 ± 0.0016 | 0.1230 ± 0.0020 | 0.1042 ± 0.0029 | 0.0982 ± 0.0125 |
| Moneyball (361616) | 99.1485 ± 148.3129 | nan | 90.5874 ± 0.9377 | 17.3321 ± 0.1964 | 20.8995 ± 0.7481 | 23.5786 ± 0.9133 | 30.7196 ± 1.3050 | 85.1957 ± 7.4612 |
| Naval | 0.0015 ± 0.0004 | nan | 0.0084 ± 0.0014 | 6159.1835 ± 40.6345 | 0.0001 ± 0.0000 | 0.0019 ± 0.0001 | 0.0020 ± 0.0001 | 0.0146 ± 0.0051 |
| Power | 4.0233 ± 0.1809 | nan | 3.8171 ± 0.1892 | 14.1800 ± 0.0787 | 3.0046 ± 0.2306 | 3.0718 ± 0.2290 | 3.6489 ± 0.1740 | 9.5650 ± 3.8317 |
| Protein | 4.3675 ± 0.0643 | nan | 3.9362 ± 0.0382 | 203262.4969 ± 828.0218 | 3.5269 ± 0.0226 | 3.3558 ± 0.0138 | 4.1510 ± 0.0265 | 7.4585 ± 1.6983 |
| QSAR_fish_toxicity (361621) | 0.9664 ± 0.0319 | nan | 1.0103 ± 0.0744 | 1.0853 ± 0.0087 | 0.8565 ± 0.0574 | 0.8851 ± 0.0628 | 0.8915 ± 0.0649 | 0.9460 ± 0.0817 |
| Wine | 0.6952 ± 0.0351 | nan | 0.6904 ± 0.0681 | 11.4637 ± 0.2424 | 0.6258 ± 0.0330 | 0.5642 ± 0.0466 | 0.6000 ± 0.0417 | 0.6852 ± 0.0342 |
| Yacht | 1.4134 ± 0.8693 | nan | 0.5512 ± 0.1457 | 6.5670 ± 0.3261 | 0.4165 ± 0.1675 | 0.8036 ± 0.4464 | 1.2210 ± 1.0873 | 3.2256 ± 1.2827 |
| abalone (361234) | 2.3291 ± 0.1477 | nan | 2.4188 ± 0.1190 | 1.1709 ± 0.0151 | 2.0641 ± 0.1218 | 2.2196 ± 0.0985 | 2.3661 ± 0.1176 | 3.0871 ± 0.3949 |
| airfoil_self_noise (361235) | 4.0643 ± 2.2717 | nan | 1.3559 ± 0.1214 | 1494.0093 ± 24.6541 | 0.9666 ± 0.1188 | 1.2715 ± 0.1672 | 1.9495 ± 0.1939 | 6.3318 ± 0.6253 |
| auction_verification (361236) | 5445.1535 ± 2783.5636 | nan | 900.4846 ± 93.2177 | 2876.2178 ± 35.1431 | 588.2404 ± 74.5416 | 496.7053 ± 97.9943 | 1122.3291 ± 117.8285 | 838.6441 ± 82.5291 |
| brazilian_houses (361267) | 516608.3000 ± 547464.2024 | nan | 16699.5304 ± 13199.5678 | 3338.5373 ± 945.3273 | 7215.3342 ± 8174.2971 | 4653.9023 ± 2336.3335 | 8861.3580 ± 6914.8962 | 9514.5037 ± 9672.7624 |
| california_housing (361255) | 55050.0398 ± 4826.4608 | nan | 45944.1189 ± 1925.5673 | 44134.7711 ± 432.1042 | 42925.4313 ± 2227.6346 | 44539.0520 ± 1954.2570 | 48372.4446 ± 1859.7897 | 74382.9958 ± 14106.7043 |
| cars (361622) | 4745.3806 ± 2259.1731 | nan | 7776.9249 ± 2947.1383 | 3441.0610 ± 87.4618 | 2045.1781 ± 162.2813 | 2383.1968 ± 260.8578 | 2123.0642 ± 188.7010 | 20568.5844 ± 13461.1258 |
| concrete_compressive_strength (361237) | 7.1753 ± 4.7777 | nan | 4.7524 ± 0.5448 | 75.9940 ± 0.8766 | 3.7765 ± 0.8155 | 4.2506 ± 0.7296 | 4.5657 ± 0.7065 | 5.6442 ± 0.5576 |
| cpu_activity (361256) | 12.2013 ± 4.1807 | nan | 2.3446 ± 0.1275 | 123282.7812 ± 1079.1185 | 2.6641 ± 0.1163 | 2.1987 ± 0.0857 | 2.6604 ± 0.3996 | 5.2785 ± 3.3704 |
| cps88wages (361261) | 7518.5296 ± 10651.3689 | nan | 405.6010 ± 47.1013 | 142.1052 ± 2.6381 | 380.4742 ± 43.3313 | 415.8071 ± 41.2163 | 380.2550 ± 43.3457 | 400.7276 ± 40.6246 |
| diamonds (361257) | 5002.4952 ± 3545.3741 | nan | 3972.5147 ± 42.3597 | 879.3667 ± 4.8997 | 529.8782 ± 18.7715 | 542.9982 ± 22.8349 | 903.0721 ± 17.9692 | 720.2492 ± 38.6047 |
| energy_efficiency (361617) | 2.1545 ± 0.9264 | nan | 0.5121 ± 0.0630 | 41.6721 ± 0.6176 | 0.4073 ± 0.0577 | 0.2876 ± 0.0674 | 0.4759 ± 0.0691 | 3.4018 ± 1.4244 |
| fifa (361272) | 243322.6148 ± 220439.6820 | nan | 18710.0942 ± 0.0000 | 0± 0 | 10351.0916 ± 1131.7528 | 9112.4963 ± 850.5738 | 9227.0417 ± 825.4616 | 12299.6116 ± 1050.3992 |
| forest_fires (361618) | 371.8608 ± 584.8017 | nan | 74.6803 ± 55.9471 | 55.6336 ± 1.5469 | 46.2842 ± 43.8281 | 60.0951 ± 37.0398 | 51.9375 ± 39.0803 | 66.7953 ± 33.5722 |
| fps_benchmark (361268) | 153.8098 ± 175.6948 | nan | 54.3875 ± 0.8100 | 270486.5750 ± 1921.2538 | 1.7752 ± 0.1797 | 1.1784 ± 0.2124 | 3.1439 ± 0.2079 | 6.4106 ± 3.0230 |
| geographical_origin_of_music (361243) | 17.4909 ± 2.7129 | nan | 18.2505 ± 1.2366 | 2.2685 ± 0.0367 | 14.3860 ± 1.7668 | 15.5055 ± 1.5756 | 15.3343 ± 1.5916 | 17.2594 ± 1.8767 |
| grid_stability (361251) | 0.0125 ± 0.0039 | nan | 0.0051 ± 0.0004 | 1.7916 ± 0.0111 | 0.0055 ± 0.0003 | 0.0082 ± 0.0003 | 0.0081 ± 0.0002 | 0.0085 ± 0.0010 |
| health_insurance (361269) | 25.3430 ± 19.4362 | nan | 18.6855 ± 0.2026 | 7.2511 ± 0.0538 | 14.4627 ± 0.1922 | 16.6180 ± 0.2839 | 14.4997 ± 0.2124 | 18.2082 ± 0.2850 |
| kin8nm (361258) | 0.0847 ± 0.0087 | nan | 0.0617 ± 0.0017 | 0.9858 ± 0.0068 | 0.0699 ± 0.0023 | 0.1222 ± 0.0032 | 0.1025 ± 0.0025 | 0.0843 ± 0.0114 |
| kings_county (361266) | 2687217.7238 ± 2712736.8872 | nan | 360455.3662 ± 28677.4921 | 36592.8902 ± 570.0643 | 117462.1375 ± 18309.4725 | 119354.3262 ± 17829.6204 | 139586.5151 ± 17340.9336 | 182138.1590 ± 28014.3590 |
| miami_housing (361260) | 437369.5516 ± 496200.5259 | nan | 95039.7730 ± 7610.7858 | 91324.1086 ± 804.7617 | 86935.5639 ± 3513.8431 | 85413.4382 ± 3605.2786 | 89039.4681 ± 3900.3201 | 109740.0256 ± 8527.6178 |
| naval_propulsion_plant (361247) | 0.0027 ± 0.0006 | nan | 0.0094 ± 0.0018 | 6573.6751 ± 60.0578 | 0.0001 ± 0.0000 | 0.0019 ± 0.0001 | 0.0020 ± 0.0001 | 0.0129 ± 0.0051 |
| physiochemical_protein (361241) | 4.9370 ± 0.1658 | nan | 3.9977 ± 0.0230 | 204121.3813 ± 1264.1073 | 3.5040 ± 0.0396 | 3.3366 ± 0.0312 | 4.0956 ± 0.0339 | 6.9638 ± 1.9365 |
| pumadyn32nh (361259) | 0.0243 ± 0.0022 | nan | 0.0364 ± 0.0006 | 19.2557 ± 0.1289 | 0.0209 ± 0.0007 | 0.0220 ± 0.0007 | 0.0211 ± 0.0006 | 0.0291 ± 0.0053 |
| red_wine (361250) | 0.7868 ± 0.1264 | nan | 0.6823 ± 0.0356 | 11.3765 ± 0.2552 | 0.6234 ± 0.0383 | 0.5521 ± 0.0429 | 0.5942 ± 0.0344 | 0.6726 ± 0.0529 |
| sarcos (361254) | 26.5323 ± 13.6386 | nan | 3.2657 ± 0.0415 | 7.8432 ± 0.0457 | 2.7871 ± 0.1564 | 2.2130 ± 0.0607 | 3.7244 ± 0.1037 | 3.4211 ± 0.2958 |
| socmob (361264) | 78.9450 ± 87.8921 | nan | 32.3465 ± 4.2748 | 10.8147 ± 0.4441 | 12.0529 ± 6.1948 | 14.3132 ± 5.9402 | 21.8913 ± 7.4460 | 29.9612 ± 9.2687 |
| solar_flare (361244) | 1.5537 ± 0.7537 | nan | 0.8749 ± 0.1761 | 0.4375 ± 0.0031 | 0.7794 ± 0.1468 | 0.9369 ± 0.0922 | 0.7973 ± 0.1330 | 0.9099 ± 0.1320 |
| space_ga (361623) | 0.1118 ± 0.0141 | nan | 0.0952 ± 0.0088 | 5387395.0000 ± 44048.0928 | 0.0918 ± 0.0182 | 0.1090 ± 0.0197 | 0.1113 ± 0.0181 | 0.1423 ± 0.0199 |
| student_performance_por (361619) | 3.2250 ± 1.0969 | nan | 3.3535 ± 0.6110 | 1.1045 ± 0.0195 | 2.6828 ± 0.5091 | 2.7174 ± 0.4920 | 2.6569 ± 0.4321 | 3.3047 ± 0.3645 |
| superconductivity (361242) | 26.1433 ± 22.0698 | nan | 34.3231 ± 0.5453 | 1101.5509 ± 6.9966 | 9.7401 ± 0.3472 | 9.2330 ± 0.2990 | 10.8765 ± 0.2820 | 14.8198 ± 1.1751 |
| video_transcoding (361252) | 98.5052 ± 79.7497 | nan | 16.2218 ± 0.4159 | 17136112.4000 ± 128141.4936 | 3.5859 ± 0.2932 | 0.8229 ± 0.1538 | 2.1865 ± 0.1244 | 1.5712 ± 0.2139 |
| wave_energy (361253) | 123541.1879 ± 87093.8173 | nan | 112344.7396 ± 780.0948 | nan | 4470.6085 ± 436.6832 | 21632.5273 ± 284.6779 | 8731.1192 ± 85.8271 | 23975.3117 ± 7464.6250 |
| white_wine (361249) | 0.7875 ± 0.0335 | nan | 0.7100 ± 0.0359 | 15.2253 ± 0.1373 | 0.6949 ± 0.0348 | 0.5811 ± 0.0257 | 0.6662 ± 0.0221 | 0.7368 ± 0.0302 |

### Test MAE
| Dataset | TabResFlow | Tabular Transformer Variational Autoencoder (TTVAE) | Thin and Deep Gaussian Processes (TDGPs) | Auto-Regressive Moving Diffusion Models (ARMD) | TabPFN | XGBoost | CatBoost | TabResNet |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| Concrete | 3.4602 ± 0.2921 | nan | 3.1324 ± 0.1791 | 51.8697 ± 0.5469 | 2.2453 ± 0.2882 | 2.7569 ± 0.2814 | 3.2815 ± 0.2722 | 4.2128 ± 0.2875 |
| Energy | 0.4254 ± 0.0695 | nan | 0.3942 ± 0.0330 | 20.1762 ± 0.3962 | 0.2856 ± 0.0337 | 0.2103 ± 0.0317 | 0.3581 ± 0.0392 | 4.1615 ± 3.0067 |
| Kin8nm | 0.0560 ± 0.0019 | nan | 0.0493 ± 0.0009 | 0.7975 ± 0.0039 | 0.0549 ± 0.0014 | 0.0960 ± 0.0018 | 0.0814 ± 0.0023 | 0.0800 ± 0.0114 |
| Moneyball (361616) | 82.3084 ± 147.9704 | nan | 72.5223 ± 0.6367 | 2.6359 ± 0.0286 | 16.4620 ± 0.7373 | 18.6725 ± 0.7344 | 24.5330 ± 1.2009 | 66.6351 ± 7.0045 |
| Naval | 0.0010 ± 0.0001 | nan | 0.0046 ± 0.0014 | 1343.1542 ± 8.0401 | 0.0001 ± 0.0000 | 0.0014 ± 0.0000 | 0.0015 ± 0.0001 | 0.0122 ± 0.0046 |
| Power | 2.8301 ± 0.0863 | nan | 2.8495 ± 0.0719 | 10.9596 ± 0.0558 | 2.0437 ± 0.0610 | 2.1288 ± 0.0702 | 2.7322 ± 0.0728 | 8.2903 ± 3.9516 |
| Protein | 2.5278 ± 0.0427 | nan | 2.8734 ± 0.0272 | 50250.1719 ± 168.7894 | 2.2714 ± 0.0235 | 2.1932 ± 0.0163 | 3.1583 ± 0.0243 | 6.1844 ± 1.4931 |
| QSAR_fish_toxicity (361621) | 0.6918 ± 0.0247 | nan | 0.7163 ± 0.0279 | 0.7369 ± 0.0041 | 0.5947 ± 0.0388 | 0.6240 ± 0.0401 | 0.6331 ± 0.0434 | 0.6768 ± 0.0347 |
| Wine | 0.4352 ± 0.0347 | nan | 0.4734 ± 0.0426 | 3.5245 ± 0.0631 | 0.4903 ± 0.0252 | 0.3671 ± 0.0344 | 0.4569 ± 0.0280 | 0.5309 ± 0.0243 |
| Yacht | 0.6382 ± 0.2814 | nan | 0.3680 ± 0.0777 | 2.0833 ± 0.0978 | 0.1873 ± 0.0633 | 0.3996 ± 0.1941 | 0.5489 ± 0.2980 | 1.7955 ± 0.7296 |
| abalone (361234) | 1.5434 ± 0.0978 | nan | 1.6484 ± 0.0687 | 0.4764 ± 0.0035 | 1.4203 ± 0.0736 | 1.5798 ± 0.0562 | 1.6822 ± 0.0698 | 2.2393 ± 0.2997 |
| airfoil_self_noise (361235) | 2.6990 ± 0.8996 | nan | 0.9856 ± 0.0917 | 429.5568 ± 4.7142 | 0.6233 ± 0.0557 | 0.8358 ± 0.0887 | 1.4429 ± 0.1265 | 5.1157 ± 0.5526 |
| auction_verification (361236) | 3284.8130 ± 1836.3193 | nan | 533.0413 ± 36.5488 | 507.1693 ± 9.3165 | 276.9586 ± 37.5523 | 224.6542 ± 38.0101 | 582.0016 ± 63.5674 | 505.6831 ± 51.5283 |
| brazilian_houses (361267) | 498789.8438 ± 527325.3125 | nan | 4020.4989 ± 558.4781 | 109.2335 ± 6.6664 | 1540.2223 ± 278.9010 | 1629.2150 ± 106.1944 | 2022.2317 ± 340.1070 | 2191.9918 ± 385.5978 |
| california_housing (361255) | 37430.1016 ± 3304.2769 | nan | 30656.4312 ± 669.9360 | 11745.2178 ± 115.2008 | 26436.8711 ± 630.3712 | 28898.4819 ± 616.9129 | 32656.5028 ± 572.7495 | 54281.4934 ± 11386.9203 |
| cars (361622) | 3197.9751 ± 1534.9386 | nan | 6332.8738 ± 2508.8185 | 869.5746 ± 14.5716 | 1384.2685 ± 108.5005 | 1613.3961 ± 178.2680 | 1520.1781 ± 118.4068 | 18323.8385 ± 13103.9223 |
| concrete_compressive_strength (361237) | 4.8940 ± 2.6834 | nan | 3.0288 ± 0.3655 | 51.7856 ± 0.5526 | 2.2283 ± 0.2765 | 2.7302 ± 0.2643 | 3.2067 ± 0.3618 | 4.2434 ± 0.3239 |
| cpu_activity (361256) | 7.7950 ± 4.5889 | nan | 1.6939 ± 0.0886 | 30161.7656 ± 216.3997 | 1.7302 ± 0.0804 | 1.5530 ± 0.0410 | 1.7882 ± 0.0516 | 4.4482 ± 3.2813 |
| cps88wages (361261) | 6963.1885 ± 10177.0049 | nan | 230.8947 ± 3.2049 | 27.9386 ± 0.1550 | 220.9764 ± 3.8062 | 249.6550 ± 5.0859 | 224.5756 ± 3.6040 | 232.3680 ± 4.1957 |
| diamonds (361257) | 3839.6118 ± 3536.7178 | nan | 2993.2968 ± 19.4141 | 122.9909 ± 0.7521 | 264.1177 ± 8.4858 | 260.2848 ± 6.9082 | 486.7339 ± 8.7350 | 354.8036 ± 19.2909 |
| energy_efficiency (361617) | 1.4448 ± 0.6024 | nan | 0.3845 ± 0.0446 | 20.1586 ± 0.3117 | 0.2771 ± 0.0315 | 0.1929 ± 0.0293 | 0.3409 ± 0.0427 | 2.8877 ± 1.3656 |
| fifa (361272) | 238415.2031 ± 221469.3750 | nan | 9889.1733 ± 0.0000 | 0± 0 | 3795.9989 ± 246.3346 | 3824.3456 ± 220.4890 | 4036.6813 ± 209.7574 | 4830.2042 ± 222.0408 |
| forest_fires (361618) | 312.9689 ± 593.3909 | nan | 21.8647 ± 8.6575 | 11.2185 ± 0.3456 | 14.9761 ± 6.9952 | 22.8329 ± 7.0027 | 20.5135 ± 8.1042 | 24.8320 ± 10.6065 |
| fps_benchmark (361268) | 133.2296 ± 176.3611 | nan | 42.4033 ± 0.5992 | 20822.2148 ± 156.5225 | 0.9875 ± 0.0388 | 0.5659 ± 0.0315 | 2.3720 ± 0.1367 | 5.2784 ± 2.6356 |
| geographical_origin_of_music (361243) | 13.4373 ± 1.5492 | nan | 14.8704 ± 0.6663 | 0.9291 ± 0.0116 | 9.7080 ± 1.1132 | 11.6030 ± 0.9604 | 11.6083 ± 0.9589 | 12.3249 ± 1.2259 |
| grid_stability (361251) | 0.0094 ± 0.0031 | nan | 0.0028 ± 0.0002 | 1.0433 ± 0.0050 | 0.0036 ± 0.0001 | 0.0060 ± 0.0001 | 0.0060 ± 0.0002 | 0.0067 ± 0.0010 |
| health_insurance (361269) | 17.7779 ± 12.4069 | nan | 17.0020 ± 0.1970 | 2.2112 ± 0.0118 | 11.1888 ± 0.2007 | 12.3405 ± 0.2230 | 11.2185 ± 0.2118 | 12.7059 ± 0.3059 |
| kin8nm (361258) | 0.0648 ± 0.0070 | nan | 0.0478 ± 0.0011 | 0.7985 ± 0.0048 | 0.0545 ± 0.0021 | 0.0957 ± 0.0023 | 0.0794 ± 0.0019 | 0.0674 ± 0.0103 |
| kings_county (361266) | 2544789.5193 ± 2749350.0709 | nan | 232827.4843 ± 3800.4805 | 2196.9810 ± 15.8372 | 60253.3985 ± 3399.2470 | 61819.0575 ± 2373.5084 | 80834.4082 ± 3718.5989 | 107033.7765 ± 4868.8705 |
| miami_housing (361260) | 347025.6875 ± 495412 | nan | 45137.8937 ± 482.9277 | 20099.2012 ± 111.6663 | 39242.1456 ± 1296.9272 | 40640.2614 ± 919.8348 | 47673.4748 ± 1125.9585 | 58181.0137 ± 7738.8015 |
| naval_propulsion_plant (361247) | 0.0021 ± 0.0005 | nan | 0.0056 ± 0.0020 | 1522.0764 ± 14.0074 | 0.0001 ± 0.0000 | 0.0014 ± 0.0000 | 0.0015 ± 0.0000 | 0.0106 ± 0.0043 |
| physiochemical_protein (361241) | 3.1271 ± 0.1105 | nan | 2.8964 ± 0.0250 | 50507.0859 ± 314.7119 | 2.2469 ± 0.0291 | 2.1821 ± 0.0234 | 3.1104 ± 0.0263 | 5.4212 ± 1.6191 |
| pumadyn32nh (361259) | 0.0189 ± 0.0010 | nan | 0.0290 ± 0.0004 | 7.0070 ± 0.0373 | 0.0166 ± 0.0006 | 0.0175 ± 0.0006 | 0.0168 ± 0.0006 | 0.0231 ± 0.0041 |
| red_wine (361250) | 0.5480 ± 0.1188 | nan | 0.4811 ± 0.0329 | 3.5151 ± 0.0543 | 0.4831 ± 0.0259 | 0.3594 ± 0.0304 | 0.4531 ± 0.0288 | 0.5158 ± 0.0519 |
| sarcos (361254) | 18.7119 ± 11.5703 | nan | 2.2486 ± 0.0417 | 3.4685 ± 0.0209 | 1.6249 ± 0.0299 | 1.2982 ± 0.0300 | 2.6317 ± 0.0669 | 2.5162 ± 0.2529 |
| socmob (361264) | 63.7669 ± 82.2721 | nan | 20.4341 ± 1.5339 | 1.2579 ± 0.0421 | 4.2007 ± 0.9983 | 5.4653 ± 1.4149 | 8.7804 ± 1.5335 | 11.8413 ± 2.9483 |
| solar_flare (361244) | 0.7060 ± 0.2141 | nan | 0.5104 ± 0.0584 | 0.2523 ± 0.0023 | 0.3698 ± 0.0604 | 0.4629 ± 0.0543 | 0.4269 ± 0.0499 | 0.4385 ± 0.0619 |
| space_ga (361623) | 0.0824 ± 0.0056 | nan | 0.0702 ± 0.0055 | 2128699.5000 ± 16259.9727 | 0.0647 ± 0.0042 | 0.0771 ± 0.0058 | 0.0802 ± 0.0055 | 0.1012 ± 0.0045 |
| student_performance_por (361619) | 2.4415 ± 0.8346 | nan | 2.5325 ± 0.3696 | 0.5617 ± 0.0039 | 1.9960 ± 0.2458 | 2.0127 ± 0.2579 | 2.0082 ± 0.2343 | 2.5507 ± 0.3290 |
| superconductivity (361242) | 21.0983 ± 21.8569 | nan | 29.2808 ± 0.6010 | 264.1196 ± 1.6294 | 5.4539 ± 0.1176 | 4.9466 ± 0.1418 | 7.0011 ± 0.1593 | 9.2058 ± 0.5307 |
| video_transcoding (361252) | 90.4064 ± 75.1337 | nan | 9.3821 ± 0.1199 | 2508689.0000 ± 18507.6113 | 0.7569 ± 0.0444 | 0.2617 ± 0.0105 | 0.9976 ± 0.0294 | 0.7278 ± 0.0839 |
| wave_energy (361253) | 95051.8203 ± 68337.7969 | nan | 88436.0550 ± 934.1824 | nan | 3020.2822 ± 81.7656 | 15653.6478 ± 212.7204 | 6730.9778 ± 80.8589 | 21608.9752 ± 7747.6771 |
| white_wine (361249) | 0.5542 ± 0.0263 | nan | 0.5206 ± 0.0251 | 5.1295 ± 0.0384 | 0.5209 ± 0.0228 | 0.3663 ± 0.0128 | 0.5171 ± 0.0124 | 0.5720 ± 0.0178 |

# TabResFlow

Training metric was NLL

Optuna optmization was done at: https://www.kaggle.com/code/ayhamo/thesis-main?scriptVersionId=242874206
for both Wine and yacht, and generlized hyperparameters were picked then for all others

(due to limited resoruces and time it takes to train TabResFlow, this was the only solution)

__Important Note:__

    1. MC samples that were generated are 1000, not 100 as in the Log!

    2. CRPS calculation was done at later stage, and MC samples were reduced to 50 for faster 
    calculations, so please ignore the results of MSE,RMSE,MAE and MAPE for the CRPS run!

## UCI

Link: https://www.kaggle.com/code/ayhamo/thesis-main/log?scriptVersionId=240458028

CRPS run: https://www.kaggle.com/code/ayhamo/thesis-main?scriptVersionId=257697815



| Dataset | Test NLL | Test CRPS | Test MSE | Test RMSE | Test MAE | Test MAPE | Original TabResFlow NLL | Original TabResFlow RMSE |
|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| Concrete | 2.8915 ± 0.1198 | 2.4417 ± 0.3120 | 27.4820 ± 5.8639 | 5.2117 ± 0.5660 | 3.4602 ± 0.2921 | 11.31% ± 1.25% | 2.90 ± 0.15 | 5.01 ± 0.70 |
| Energy | 0.6444 ± 0.1246 | 0.2943 ± 0.0322 | 0.4434 ± 0.2697 | 0.6453 ± 0.1645 | 0.4254 ± 0.0695 | 1.95% ± 0.27% | 0.77 ± 0.19 | 1.45 ± 2.24 |
| Kin8nm | -1.2842 ± 0.0320 | 0.0387 ± 0.0013 | 0.0053 ± 0.0004 | 0.0729 ± 0.0030 | 0.0560 ± 0.0019 | 10.36% ± 0.63% | -1.29 ± 0.04 | 0.07 ± 0.00 |
| Naval | -5.3268 ± 0.1114 | 0.0007 ± 0.0001 | 0.0000 ± 0.0000 | 0.0015 ± 0.0004 | 0.0010 ± 0.0001 | 0.10% ± 0.01% | -5.30 ± 0.11 | 0.00 ± 0.00 |
| Power | 2.5808 ± 0.0220 | 1.9334 ± 0.0505 | 16.2194 ± 1.4487 | 4.0233 ± 0.1809 | 2.8301 ± 0.0863 | 0.62% ± 0.02% | 2.60 ± 0.04 | 3.98 ± 0.19 |
| Protein | 1.9293 ± 0.0184 | 1.6971 ± 0.0168 | 19.0796 ± 0.5625 | 4.3675 ± 0.0643 | 2.5278 ± 0.0427 | 11.06% ± 1.43% | 1.95 ± 0.04 | 4.44 ± 0.10 |
| Wine | -1.1610 ± 0.1092 | 0.3172 ± 0.0216 | 0.4846 ± 0.0479 | 0.6952 ± 0.0351 | 0.4352 ± 0.0347 | 7.87% ± 0.71% | -0.85 ± 0.27 | 0.39 ± 0.06 |
| Yacht | 0.5558 ± 0.2196 | 0.5382 ± 0.2553 | 2.7534 ± 3.8523 | 1.4134 ± 0.8693 | 0.6382 ± 0.2814 | 30.19% ± 25.13% | 0.67 ± 0.32 | 0.47 ± 0.11 |

## OpenML-CTR23

First 2 datasets Link: https://www.kaggle.com/code/ayhamo/thesis-main?scriptVersionId=242940634

The rest of datasets: https://www.kaggle.com/code/ayhamo/thesis-main?scriptVersionId=243023151

CRPS run 1 (till naval_plant): https://www.kaggle.com/code/ayhamo/thesis-main?scriptVersionId=257697847

CRPS run 2 (from white_wine): https://www.kaggle.com/code/ayhamo/thesis-main?scriptVersionId=257844817


| Dataset | Test NLL | Test CRPS | Test MSE | Test RMSE | Test MAE | Test MAPE |
|---------|---------|---------|---------|---------|---------|---------|
| grid_stability (361251) | -3.0600 ± 0.3524| 0.0065 ± 0.0017 | 0.0002 ± 0.0001 | 0.0125 ± 0.0039 | 0.0094 ± 0.0031 | 135.87% ± 62.57% |
| video_transcoding (361252) | 4.8826 ± 1.8184 | 16.6803 ± 20.9636 | 16063.2969 ± 19043.5284 | 98.5052 ± 79.7497 | 90.4064 ± 75.1337 | 3762.06% ± 3321.41% |
| wave_energy (361253) | 12.1951 ± 1.2146| 231868.0315 ± 463776.9219 | 22847758336 ± 31917230080 | 123541.1879 ± 87093.8173 | 95051.8203 ± 68337.7969 | 2.52% ± 1.79% |
| sarcos (361254) | 4.1661 ± 0.8536 | 15.5040 ± 20.0312 | 889.9745 ± 769.7089 | 26.5323 ± 13.6386 | 18.7119 ± 11.5703 | 432.32% ± 323.20% |
| california_housing (361255) | 12.0207 ± 0.1364| 25868.2331 ± 1542.1326 | 3053801728 ± 551256192 | 55050.0398 ± 4826.4608 | 37430.1016 ± 3304.2769 | 20.76% ± 2.33% |
| cpu_activity (361256) | 2.9801 ± 0.4061 | 4.2195 ± 4.5878 | 166.3508 ± 104.4023 | 12.2013 ± 4.1807 | 7.7950 ± 4.5889 | 13.67% ± 9.47% |
| diamonds (361257) | 8.4006 ± 0.8528 | 3516.9083 ± 5325.3385 | 37594632 ± 46169312 | 5002.4952 ± 3545.3741 | 3839.6118 ± 3536.7178 | 247.06% ± 253.41% |
| kin8nm (361258) | -1.1487 ± 0.0845| 0.0445 ± 0.0044 | 0.0072 ± 0.0016 | 0.0847 ± 0.0087 | 0.0648 ± 0.0070 | 12.33% ± 1.32% |
| pumadyn32nh (361259) | -2.3248 ± 0.0482| 0.0127 ± 0.0004 | 0.0006 ± 0.0001 | 0.0243 ± 0.0022 | 0.0189 ± 0.0010 | 293.41% ± 78.34% |
| miami_housing (361260) | 12.9735 ± 0.7088| 226725.1161 ± 240388.6019 | 437507096576 ± 868867506176 | 437369.5516 ± 496200.5259 | 347025.6875 ± 495412 | 112.97% ± 172.29% |
| cps88wages (361261) | 7.9640 ± 1.1164 | 1771.2695 ± 1798.8748 | 169979936 ± 378400000 | 7518.5296 ± 10651.3689 | 6963.1885 ± 10177.0049 | 1957.35% ± 2854.53% |
| socmob (361264) | 4.1575 ± 0.7145 | 23.4827 ± 21.4010 | 13957.3438 ± 27060.8379 | 78.9450 ± 87.8921 | 63.7669 ± 82.2721 | 2462.06% ± 3368.49% |
| kings_county (361266) | 14.8254 ± 0.8506| 2566143.5319 ± 3258756.7758 | 14580080379298.2305 ± 21983366501781.0586 | 2687217.7238 ± 2712736.8872 | 2544789.5193 ± 2749350.0709 | 639.58% ± 692.30% |
| brazilian_houses (361267) | 12.4966 ± 1.7772| 273592.3841 ± 475166.7514 | 566601252864 ± 980091994112 | 516608.3000 ± 547464.2024 | 498789.8438 ± 527325.3125 | 17260.53% ± 18186.37% |
| fps_benchmark (361268) | 5.9665 ± 1.3995 | 29.9009 ± 57.5754 | 54526.1332 ± 122610.6687 | 153.8098 ± 175.6948 | 133.2296 ± 176.3611 | 130.27% ± 167.01% |
| health_insurance (361269) | 3.9363 ± 0.3288 | 9.0640 ± 0.9230 | 1020.0330 ± 1959.5533 | 25.3430 ± 19.4362 | 17.7779 ± 12.4069 | 51.02% ± 41.59% |
| fifa (361272) | 14.4376 ± 2.7050| 90946.3945 ± 78775.3609 | 107799552000 ± 199156842496 | 243322.6148 ± 220439.6820 | 238415.2031 ± 221469.3750 | 14993.53% ± 13468.27% |
| abalone (361234) | 1.9358 ± 0.0424 | 1.0597 ± 0.0524 | 5.4463 ± 0.6886 | 2.3291 ± 0.1477 | 1.5434 ± 0.0978 | 14.30% ± 0.52% |
| airfoil_self_noise (361235) | 2.6113 ± 0.3220 | 1.7791 ± 0.3934 | 21.6791 ± 30.4767 | 4.0643 ± 2.2717 | 2.6990 ± 0.8996 | 2.16% ± 0.73% |
| auction_verification (361236) | 9.1342 ± 0.3532 | 1864.6033 ± 1120.3322 | 37397924 ± 35045820 | 5445.1535 ± 2783.5636 | 3284.8130 ± 1836.3193 | 438.13% ± 329.91% |
| concrete_compressive_strength (361237)| 3.1155 ± 0.2340 | 3.0754 ± 0.4363 | 74.3113 ± 128.0850 | 7.1753 ± 4.7777 | 4.8940 ± 2.6834 | 15.77% ± 7.75% |
| physiochemical_protein (361241) | 2.2091 ± 0.0383 | 2.1946 ± 0.1644 | 24.4014 ± 1.6434 | 4.9370 ± 0.1658 | 3.1271 ± 0.1105 | 53.12% ± 3.89% |
| superconductivity (361242) | 4.0911 ± 0.7489 | 11.3786 ± 14.5353 | 1170.5513 ± 2171.2662 | 26.1433 ± 22.0698 | 21.0983 ± 21.8569 | 2523.67% ± 3108.45% |
| geographical_origin_of_music (361243)| 4.0810 ± 0.1098 | 13.2288 ± 13.1600 | 313.2901 ± 106.4647 | 17.4909 ± 2.7129 | 13.4373 ± 1.5492 | 109.65% ± 26.40% |
| solar_flare (361244) | 1.0424 ± 0.4577 | 0.6500 ± 0.3381 | 2.9819 ± 2.8886 | 1.5537 ± 0.7537 | 0.7060 ± 0.2141 | 116.62% ± 50.88% |
| naval_propulsion_plant (361247) | -4.5097 ± 0.2364| 0.0017 ± 0.0005 | 0 ± 0 | 0.0027 ± 0.0006 | 0.0021 ± 0.0005 | 0.22% ± 0.06% |
| white_wine (361249) | 0.6118 ± 0.2919 | 0.3997 ± 0.0170 | 0.6213 ± 0.0533 | 0.7875 ± 0.0335 | 0.5542 ± 0.0263 | 9.72% ± 0.68% |
| red_wine (361250) | 0.4910 ± 0.3961 | 0.3855 ± 0.0194 | 0.6350 ± 0.2270 | 0.7868 ± 0.1264 | 0.5480 ± 0.1188 | 9.79% ± 1.89% |
| Moneyball (361616) | 5.1378 ± 0.6119 | 39.3231 ± 31.8497 | 31827.1191 ± 82224.6953 | 99.1485 ± 148.3129 | 82.3084 ± 147.9704 | 11.84% ± 21.03% |
| energy_efficiency (361617) | 1.8445 ± 0.4596 | 1.1639 ± 0.4974 | 5.5001 ± 4.4427 | 2.1545 ± 0.9264 | 1.4448 ± 0.6024 | 6.75% ± 2.50% |
| forest_fires (361618) | 5.6332 ± 0.5838 | 69.4872 ± 64.3380 | 480273.5938 ± 1284821.1250 | 371.8608 ± 584.8017 | 312.9689 ± 593.3909 | 13310.74% ± 26500.23% |
| student_performance_por (361619) | 2.5962 ± 0.3028 | 2.3243 ± 2.3020 | 11.6036 ± 9.4291 | 3.2250 ± 1.0969 | 2.4415 ± 0.8346 | 20.55% ± 9.69% |
| QSAR_fish_toxicity (361621) | 1.3152 ± 0.0720 | 0.4836 ± 0.0403 | 0.9350 ± 0.0616 | 0.9664 ± 0.0319 | 0.6918 ± 0.0247 | 26.30% ± 9.24% |
| cars (361622) | 9.3759 ± 0.4065 | 1704.8845 ± 935.2053 | 27622500 ± 21774774 | 4745.3806 ± 2259.1731 | 3197.9751 ± 1534.9386 | 15.50% ± 8.33% |
| space_ga (361623) | -0.8686 ± 0.0672| 0.0642 ± 0.0076 | 0.0127 ± 0.0035 | 0.1118 ± 0.0141 | 0.0824 ± 0.0056 | 16.20% ± 1.48% |

# Tabular Transformer Variational Autoencoder (TTVAE)

will be updated


# Thin and Deep Gaussian Processes (TDGPs)

Installation is explained in requirements.txt and must be followed in order, otherwise it won't work!

- They are commented because they use old versions of libraries and they would conflict with all other libraries, which is why they must be installed only when TDGPs will be used

- after usage the environment must be re-initialized to use the latest version for other models!

Why regression metrics are computed on the unscaled target
- Metrics like RMSE, MAE, R² are easier to interpret in the original units of the target variable.


## UCI

Link all (without yacht): https://www.kaggle.com/code/ayhamo/thesis-main/log?scriptVersionId=260279644

Link yacht: https://www.kaggle.com/code/ayhamo/thesis-main?scriptVersionId=260324501

Note: the NLL was unscaled later, and NLL in the above code is only scaled, it is explained below.

| Dataset | Test NLL | Test CRPS | Test MSE | Test RMSE | Test MAE | Test MAPE |
|---|---|---|---|---|---|---|
| Concrete | 3.718 ± 0.4709 | 2.4322 ± 0.1734 | 21.7218 ± 4.8990 | 4.6327 ± 0.5094 | 3.1324 ± 0.1791 | 11.10% ± 0.69% |
| Energy | 1.1777 ± 0.0252 | 0.3639 ± 0.0155 | 0.2915 ± 0.0588 | 0.5371 ± 0.0553 | 0.3942 ± 0.0330 | 1.92% ± 0.13% |
| Kin8nm | -0.7756 ± 0.0734 | 0.0372 ± 0.0008 | 0.0041 ± 0.0002 | 0.0636 ± 0.0014 | 0.0493 ± 0.0009 | 9.21% ± 0.53% |
| Naval | 4.8496 ± 3.5978 | 0.0043 ± 0.0012 | 0.0001 ± 0.0000 | 0.0084 ± 0.0014 | 0.0046 ± 0.0014 | 0.47% ± 0.14% |
| Power | 3.727  ± 0.2112 | 2.1911 ± 0.0664 | 14.6059 ± 1.4419 | 3.8171 ± 0.1892 | 2.8495 ± 0.0719 | 0.63% ± 0.02% |
| Protein | 10.4006 ± 0.2014 | 2.4666 ± 0.0265 | 15.4953 ± 0.3015 | 3.9362 ± 0.0382 | 2.8734 ± 0.0272 | 62.87% ± 1.62% |
| Wine | 6.0745 ± 1.2653 | 0.4034 ± 0.0416 | 0.4812 ± 0.0949 | 0.6904 ± 0.0681 | 0.4734 ± 0.0426 | 8.83% ± 0.93% |
| Yacht | 1.4683 ± 0.0315 | 0.4563 ± 0.0278 | 0.3250 ± 0.1653 | 0.5512 ± 0.1457 | 0.3680 ± 0.0777 | 73.57% ± 55.88% |

* NLL_corrected = NLL_reported_scaled + log(σ), std stays the same since spread is also the same, here is the correction table:


## OpenML-CTR23

Link 1 (till cpu_activity): https://www.kaggle.com/code/ayhamo/thesis-main?scriptVersionId=260279679

Link 2 (till socmob): https://www.kaggle.com/code/ayhamo/thesis-main?scriptVersionId=260328175

Link 3 (fps_benchmark and health_insurance): https://www.kaggle.com/code/ayhamo/thesis-main/edit/run/259113493

Link 4 (brazilian_houses, 3 folds): https://www.kaggle.com/code/ayhamo/thesis-main/log?scriptVersionId=260474999

Link 5 (fifa, 1 fold): https://www.kaggle.com/code/ayhamo/thesis-main?scriptVersionId=260474962

*All Fifa folds crashes the program, but first fold, so std of this is 0

Link 6 (abalone to superconductivity): https://www.kaggle.com/code/ayhamo/thesis-main/log?scriptVersionId=260470673

Link 7 (origin_of_music & solar_flare): https://www.kaggle.com/code/ayhamalt/thesis/log?scriptVersionId=260716124

Link 8 (from naval to forest_fires): https://www.kaggle.com/code/ayhamalt/thesis?scriptVersionId=260915496

Link 9 (from student_performance to space_ga[end]): https://www.kaggle.com/code/ayhamalt/thesis?scriptVersionId=260997427



| Dataset | Test NLL | Test CRPS | Test MSE | Test RMSE | Test MAE | Test MAPE |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| grid_stability (361251) | -3.9017 ± 0.0922 | 0.0024 ± 0.0002 | 0.0000 ± 0.0000 | 0.0051 ± 0.0004 | 0.0028 ± 0.0002 | 23.19% ± 3.01% |
| video_transcoding (361252) | 53.0538 ± 2.8714 | 8.5438 ± 0.1180 | 263.3198 ± 13.5337 | 16.2218 ± 0.4159 | 9.3821 ± 0.1199 | 318.24% ± 9.05% |
| wave_energy (361253) | 60.2542 ± 0.7758 | 82569.0760 ± 913.4778 | 12621949056.0449 ± 175402380.8487 | 112344.7396 ± 780.0948 | 88436.0550 ± 934.1824 | 2.35% ± 0.02% |
| sarcos (361254) | 2.7202 ± 0.0235 | 1.6973 ± 0.0302 | 10.6668 ± 0.2716 | 3.2657 ± 0.0415 | 2.2486 ± 0.0417 | 52.38% ± 6.24% |
| california_housing (361255) | 14.7245 ± 0.3054 | 24902.5952 ± 618.9691 | 2114569869.4119 ± 176134436.5426 | 45944.1189 ± 1925.5673 | 30656.4312 ± 669.9360 | 17.29% ± 0.36% |
| cpu_activity (361256) | 2.2650 ± 0.0594 | 1.2421 ± 0.0610 | 5.5134 ± 0.6063 | 2.3446 ± 0.1275 | 1.6939 ± 0.0886 | 2.31% ± 0.16% |
| diamonds (361257) | 58.7968 ± 1.2763 | 2785.3115 ± 19.9483 | 15782666.9924 ± 337309.1582 | 3972.5147 ± 42.3597 | 2993.2968 ± 19.4141 | 182.63% ± 3.33% |
| kin8nm (361258) | -0.8852 ± 0.0929 | 0.0359 ± 0.0010 | 0.0038 ± 0.0002 | 0.0617 ± 0.0017 | 0.0478 ± 0.0011 | 9.06% ± 0.57% |
| pumadyn32nh (361259) | 46.8111 ± 1.7536 | 0.0271 ± 0.0004 | 0.0013 ± 0.0000 | 0.0364 ± 0.0006 | 0.0290 ± 0.0004 | 103.37% ± 2.34% |
| miami_housing (361260) | 12.6450 ± 0.1490 | 35180.3496 ± 527.1171 | 9090482510.0571 ± 1456990785.1374 | 95039.7730 ± 7610.7858 | 45137.8937 ± 482.9277 | 11.03% ± 0.29% |
| cps88wages (361261) | 23.5202 ± 3.2308 | 201.8928 ± 2.8695 | 166730.7228 ± 40075.5367 | 405.6010 ± 47.1013 | 230.8947 ± 3.2049 | 54.67% ± 1.82% |
| socmob (361264) | 32.8790 ± 8.0524 | 18.2049 ± 1.5463 | 1064.5726 ± 266.0666 | 32.3465 ± 4.2748 | 20.4341 ± 1.5339 | 734.52% ± 84.41% |
| kings_county (361266) | 62.0491 ± 10.2573 | 214278.6648 ± 4125.2262 | 130750469599.8554 ± 21531718468.4040 | 360455.3662 ± 28677.4921 | 232827.4843 ± 3800.4805 | 53.14% ± 1.08% |
| brazilian_houses (361267) | 1037.9037 ± 1433.8985 | 3360.5704 ± 836.8226 | 453149611.6628 ± 548600000.0000 | 16699.5304 ± 13199.5678 | 4020.4989 ± 558.4781 | 117.83% ± 8.19% |
| fps_benchmark (361268) | 52.2627 ± 1.6689 | 39.5227 ± 0.5548 | 2958.6520 ± 88.0224 | 54.3875 ± 0.8100 | 42.4033 ± 0.5992 | 41.93% ± 0.95% |
| health_insurance (361269) | 51.3758 ± 1.1180 | 15.9719 ± 0.1980 | 349.6880 ± 7.7430 | 18.6855 ± 0.2026 | 17.0020 ± 0.1970 | 42.20% ± 1.06% |
| fifa (361272) | 8932.0272 ± 0.0000 | 60.3657 ± 0.0000 | 350067624.6939 ± 0.0000 | 18710.0942 ± 0.0000 | 9889.1733 ± 0.0000 | 498.28% ± 0.00% |
| abalone (361234) | 9.4121 ± 1.2326 | 1.3956 ± 0.0625 | 5.8646 ± 0.5809 | 2.4188 ± 0.1190 | 1.6484 ± 0.0687 | 16.12% ± 0.20% |
| airfoil_self_noise (361235) | 3.7654 ± 0.1458 | 0.7271 ± 0.0659 | 1.8531 ± 0.3224 | 1.3559 ± 0.1214 | 0.9856 ± 0.0917 | 0.79% ± 0.07% |
| auction_verification (361236) | 17.5138 ± 0.0469 | 484.1481 ± 30.6753 | 819562.0248 ± 169028.9401 | 900.4846 ± 93.2177 | 533.0413 ± 36.5488 | 50.54% ± 14.93% |
| concrete_compressive_strength (361237) | 6.4571 ± 0.3688 | 2.3489 ± 0.3038 | 22.8821 ± 5.1974 | 4.7524 ± 0.5448 | 3.0288 ± 0.3655 | 10.10% ± 1.20% |
| physiochemical_protein (361241) | 11.7330 ± 0.3020 | 2.4803 ± 0.0254 | 15.9822 ± 0.1843 | 3.9977 ± 0.0230 | 2.8964 ± 0.0250 | 64.03% ± 2.65% |
| superconductivity (361242) | 55.9570 ± 1.6135 | 27.4435 ± 0.5922 | 1178.3707 ± 37.2792 | 34.3231 ± 0.5453 | 29.2808 ± 0.6010 | 4905.67% ± 5732.11% |
| geographical_origin_of_music (361243) | 53.5667 ± 7.5850 | 13.8535 ± 0.6738 | 334.6108 ± 46.1712 | 18.2505 ± 1.2366 | 14.8704 ± 0.6663 | 164.77% ± 24.87% |
| solar_flare (361244) | 57.5413 ± 26.6704 | 0.4636 ± 0.0595 | 0.7965 ± 0.3175 | 0.8749 ± 0.1761 | 0.5104 ± 0.0584 | 77.37% ± 2.66% |
| naval_propulsion_plant (361247) | 2.9849 ± 5.0225 | 0.0052 ± 0.0018 | 0.0001 ± 0.0000 | 0.0094 ± 0.0018 | 0.0056 ± 0.0020 | 0.57% ± 0.20% |
| white_wine (361249) | 4.8953 ± 0.7340 | 0.4328 ± 0.0237 | 0.5054 ± 0.0524 | 0.7100 ± 0.0359 | 0.5206 ± 0.0251 | 9.23% ± 0.55% |
| red_wine (361250) | 5.7031 ± 0.9420 | 0.4082 ± 0.0314 | 0.4668 ± 0.0494 | 0.6823 ± 0.0356 | 0.4811 ± 0.0329 | 8.87% ± 0.65% |
| Moneyball (361616) | 56.5110 ± 1.1218 | 67.6429 ± 0.5373 | 8206.9628 ± 169.9284 | 90.5874 ± 0.9377 | 72.5223 ± 0.6367 | 10.32% ± 0.26% |
| energy_efficiency (361617) | 3.4767 ± 0.0293 | 0.3562 ± 0.0186 | 0.2662 ± 0.0643 | 0.5121 ± 0.0630 | 0.3845 ± 0.0446 | 1.92% ± 0.31% |
| forest_fires (361618) | 202.2548 ± 247.9115 | 19.0932 ± 9.0204 | 8707.2220 ± 9348.0642 | 74.6803 ± 55.9471 | 21.8647 ± 8.6575 | 515.64% ± 233.10% |
| student_performance_por (361619) | 58.3599 ± 22.5691 | 2.3706 ± 0.3717 | 11.6190 ± 4.0877 | 3.3535 ± 0.6110 | 2.5325 ± 0.3696 | 18.89% ± 2.27% |
| QSAR_fish_toxicity (361621) | 5.7866 ± 0.8930 | 0.5936 ± 0.0266 | 1.0263 ± 0.1517 | 1.0103 ± 0.0744 | 0.7163 ± 0.0279 | 25.13% ± 2.25% |
| cars (361622) | 51.9526 ± 18.3122 | 5842.3305 ± 2434.2272 | 69166185.4658 ± 35316919.5792 | 7776.9249 ± 2947.1383 | 6332.8738 ± 2508.8185 | 33.32% ± 13.60% |
| space_ga (361623) | 1.4078 ± 0.6520 | 0.0581 ± 0.0048 | 0.0091 ± 0.0017 | 0.0952 ± 0.0088 | 0.0702 ± 0.0055 | 14.40% ± 1.66% |

# Auto-Regressive Moving Diffusion Models (ARMD)


## UCI

Link: https://www.kaggle.com/code/ayhamo/thesis-main?scriptVersionId=262006229

| Dataset | Test NLL | Test CRPS | Test MSE | Test RMSE | Test MAE | Test MAPE |
|:---|:---|:---|:---|:---|:---|:---|
| Concrete | 51.8697 ± 0.5469 | 28941764.7000 ± 645085.9229 | 5788.3535 ± 129.0174 | 76.0765 ± 0.8481 | 51.8697 ± 0.5469 | 372931018.75% ± 6822817.83% |
| Energy | 20.1762 ± 0.3962 | 8692267.9750 ± 349840.2804 | 1738.4543 ± 69.9680 | 41.6863 ± 0.8423 | 20.1762 ± 0.3962 | 35.87% ± 0.60% |
| Kin8nm | 0.7975 ± 0.0039 | 4844.7940 ± 51.9951 | 0.9697 ± 0.0104 | 0.9847 ± 0.0053 | 0.7975 ± 0.0039 | 505.85% ± 140.68% |
| Naval | 1343.1542 ± 8.0401 | 189685965619.2000 ± 2510031123.3265 | 37937192.0000 ± 502006.6562 | 6159.1835 ± 40.6345 | 1343.1542 ± 8.0401 | 10229797.03% ± 291639.89% |
| Power | 10.9596 ± 0.0558 | 1005387.5375 ± 11128.2658 | 201.0782 ± 2.2256 | 14.1800 ± 0.0787 | 10.9596 ± 0.0558 | 19.76% ± 0.10% |
| Protein | 50250.1692 ± 168.7890 | 206581632126156.8125 ± 1679975204008.8855 | 41316327424.0000 ± 335995072.0000 | 203262.4969 ± 828.0218 | 50250.1719 ± 168.7894 | 24337.78% ± 9465.64% |
| Wine | 3.5245 ± 0.0631 | 657373.1438 ± 27750.8696 | 131.4754 ± 5.5502 | 11.4637 ± 0.2424 | 3.5245 ± 0.0631 | 43.79% ± 0.58% |
| Yacht | 2.0833 ± 0.0978 | 216154.6469 ± 21596.0144 | 43.2317 ± 4.3192 | 6.5670 ± 0.3261 | 2.0833 ± 0.0978 | 317.77% ± 65.95% |


## OpenML-CTR23

Link 1 (till health_insurance): https://www.kaggle.com/code/ayhamo/thesis-main?scriptVersionId=262006292

Link 2 (from abalone till end): https://www.kaggle.com/code/ayhamo/thesis-main/log?scriptVersionId=262226003


| Dataset | Test NLL | Test CRPS | Test MSE | Test RMSE | Test MAE | Test MAPE |
|---|---|---|---|---|---|---|
| grid_stability (361251) | 16046.4731 ± 198.0041 | 1.0433 ± 0.0050 | 3.2100 ± 0.0396 | 1.7916 ± 0.0111 | 1.0433 ± 0.0050 | 101.45% ± 3.76% |
| video_transcoding (361252) | 1468313927389334784.0000 ± 21901444588499712.0000 | 2508689.0727 ± 18507.5873 | 293662751719424.0000 ± 4380295954432.0000 | 17136112.4000 ± 128141.4936 | 2508689.0000 ± 18507.6113 | 278.61% ± 2.29% |
| wave_energy(361253) | 0 ± 0 | 0± 0 | 0± 0 | 0± 0 | 0± 0 | 0% ± 0% |
| sarcos (361254) | 307589.5938 ± 3577.3373 | 3.4685 ± 0.0209 | 61.5187 ± 0.7155 | 7.8432 ± 0.0457 | 3.4685 ± 0.0209 | 9875.24% ± 830.98% |
| california_housing (361255) | 9740323546726.4004 ± 190125207511.4850 | 11745.2175 ± 115.2010 | 1948064768.0000 ± 38024968.0000 | 44134.7711 ± 432.1042 | 11745.2178 ± 115.2008 | 79.59% ± 2.25% |
| cpu_activity (361256) | 75999042810675.2031 ± 1332833129912.8777 | 30161.7643 ± 216.4013 | 15199808512.0000 ± 266567056.0000 | 123282.7812 ± 1079.1185 | 30161.7656 ± 216.3997 | 201778.04% ± 80435.29% |
| diamonds (361257) | 3866549376.0000 ± 43094511.0564 | 122.9909 ± 0.7521 | 773309.8750 ± 8618.9453 | 879.3667 ± 4.8997 | 122.9909 ± 0.7521 | 55.80% ± 0.21% |
| kin8nm (361258) | 4855.8146 ± 66.9399 | 0.7985 ± 0.0048 | 0.9719 ± 0.0134 | 0.9858 ± 0.0068 | 0.7985 ± 0.0048 | 552.37% ± 150.06% |
| pumadyn32nh (361259) | 1853991.4500 ± 24740.8613 | 7.0070 ± 0.0373 | 370.7990 ± 4.9482 | 19.2557 ± 0.1289 | 7.0070 ± 0.0373 | 292.54% ± 38.31% |
| miami_housing (361260) | 41703703786291.2031 ± 734861509716.5266 | 20099.2022 ± 111.6660 | 8340740608.0000 ± 146972224.0000 | 91324.1086 ± 804.7617 | 20099.2012 ± 111.6663 | 40376742.50% ± 638986.57% |
| cps88wages (361261) | 101004191.2000 ± 3765797.4712 | 27.9386 ± 0.1550 | 20200.8398 ± 753.1591 | 142.1052 ± 2.6381 | 27.9386 ± 0.1550 | 63.13% ± 0.15% |
| socmob (361264) | 585770.1406 ± 48051.2615 | 1.2579 ± 0.0421 | 117.1548 ± 9.6103 | 10.8147 ± 0.4441 | 1.2579 ± 0.0421 | 222.21% ± 10.55% |
| kings_county (361266) | 6696823265689.5996 ± 207414993863.7949 | 2196.9810 ± 15.8375 | 1339364608.0000 ± 41483080.0000 | 36592.8902 ± 570.0643 | 2196.9810 ± 15.8372 | 230094020.00% ± 2568835.30% |
| brazilian_houses (361267) | 60197377024.0000 ± 25986399891.2135 | 109.2335 ± 6.6664 | 12039475.0000 ± 5197279.5000 | 3338.5373 ± 945.3273 | 109.2335 ± 6.6664 | 15645422.42% ± 3995110.80% |
| fps_benchmark (361268) | 365833405805363.1875 ± 5195003905751.0010 | 20822.2123 ± 156.5238 | 73166684160.0000 ± 1039000896.0000 | 270486.5750 ± 1921.2538 | 20822.2148 ± 156.5225 | 39.89% ± 0.11% |
| health_insurance (361269) | 262902.6141 ± 3903.3655 | 2.2112 ± 0.0118 | 52.5813 ± 0.7807 | 7.2511 ± 0.0538 | 2.2112 ± 0.0118 | 67.98% ± 1.93% |
| fifa (361272) | 0 ± 0 | 0± 0 | 0± 0 | 0± 0 | 0± 0 | 0% ± 0% |
| abalone (361234) | 0.4764 ± 0.0035 | 6852.1713 ± 177.3544 | 1.3712 ± 0.0355 | 1.1709 ± 0.0151 | 0.4764 ± 0.0035 | 91.30% ± 2.98% |
| airfoil_self_noise (361235) | 429.5568 ± 4.7142 | 11163358617.6000 ± 365914748.0158 | 2232671.7500 ± 73182.9297 | 1494.0093 ± 24.6541 | 429.5568 ± 4.7142 | 133.99% ± 2.23% |
| auction_verification (361236) | 507.1693 ± 9.3165 | 41369319424.0000 ± 1017404228.8739 | 8273864.0000 ± 203480.6875 | 2876.2178 ± 35.1431 | 507.1693 ± 9.3165 | 175.22% ± 7.63% |
| concrete_compressive_strength (361237) | 51.7856 ± 0.5526 | 28879276.8000 ± 666461.1526 | 5775.8564 ± 133.2921 | 75.9940 ± 0.8766 | 51.7856 ± 0.5526 | 166397506.25% ± 5618180.66% |
| physiochemical_protein (361241) | 50507.0864 ± 314.7118 | 208335671604019.1875 ± 2586883697441.3501 | 41667137536.0000 ± 517376928.0000 | 204121.3813 ± 1264.1073 | 50507.0859 ± 314.7119 | 28369.21% ± 6172.53% |
| superconductivity (361242) | 264.1196 ± 1.6294 | 6067316633.6000 ± 77265609.9660 | 1213463.3750 ± 15453.0752 | 1101.5509 ± 6.9966 | 264.1196 ± 1.6294 | 34496541.56% ± 1055591.46% |
| geographical_origin_of_music (361243) | 0.9291 ± 0.0116 | 25733.6937 ± 835.3423 | 5.1475 ± 0.1671 | 2.2685 ± 0.0367 | 0.9291 ± 0.0116 | 354.92% ± 24.07% |
| solar_flare (361244) | 0.2523 ± 0.0023 | 953.5112 ± 13.5871 | 0.1914 ± 0.0027 | 0.4375 ± 0.0031 | 0.2523 ± 0.0023 | 38.08% ± 0.31% |
| naval_propulsion_plant (361247) | 1522.0764 ± 14.0074 | 216084057292.8000 ± 3938915264.8445 | 43216808.0000 ± 787784.8125 | 6573.6751 ± 60.0578 | 1522.0764 ± 14.0074 | 11543966.25% ± 275775.45% |
| white_wine (361249) | 5.1295 ± 0.0384 | 1159133.4000 ± 20821.1927 | 231.8274 ± 4.1642 | 15.2253 ± 0.1373 | 5.1295 ± 0.0384 | 34.78% ± 0.23% |
| red_wine (361250) | 3.5151 ± 0.0543 | 647448.3562 ± 29165.2181 | 129.4904 ± 5.8330 | 11.3765 ± 0.2552 | 3.5151 ± 0.0543 | 44.04% ± 0.81% |
| Moneyball (361616) | 2.6359 ± 0.0286 | 1502193.3875 ± 34019.5532 | 300.4394 ± 6.8039 | 17.3321 ± 0.1964 | 2.6359 ± 0.0286 | 23.90% ± 0.15% |
| energy_efficiency (361617) | 20.1586 ± 0.3117 | 8684711.6500 ± 257250.0924 | 1736.9431 ± 51.4500 | 41.6721 ± 0.6176 | 20.1586 ± 0.3117 | 35.60% ± 0.42% |
| forest_fires (361618) | 11.2185 ± 0.3456 | 15487436.6000 ± 862946.9561 | 3097.4878 ± 172.5894 | 55.6336 ± 1.5469 | 11.2185 ± 0.3456 | 94.34% ± 14.96% |
| student_performance_por (361619) | 0.5617 ± 0.0039 | 6097.2813 ± 216.5593 | 1.2202 ± 0.0433 | 1.1045 ± 0.0195 | 0.5617 ± 0.0039 | 44.31% ± 0.35% |
| QSAR_fish_toxicity (361621) | 0.7369 ± 0.0041 | 5886.5473 ± 93.6446 | 1.1780 ± 0.0187 | 1.0853 ± 0.0087 | 0.7369 ± 0.0041 | 82.97% ± 2.98% |
| cars (361622) | 869.5746 ± 14.5716 | 59242753228.8000 ± 2997495578.6556 | 11848550.0000 ± 599499.0625 | 3441.0610 ± 87.4618 | 869.5746 ± 14.5716 | 47.61% ± 1.00% |
| space_ga (361623) | 2128699.3379 ± 16259.9948 | 145129825995902144.0000 ± 2368378464024677.0000 | 29025963606016.0000 ± 473676480512.0000 | 5387395.0000 ± 44048.0928 | 2128699.5000 ± 16259.9727 | 15.99% ± 0.19% |


# TabPFN

Training metric was RMSE

TabPFN hyperparameters are optimized automatically during the training.

## UCI

Link: https://www.kaggle.com/code/ayhamo/thesis-main?scriptVersionId=241417558

| Dataset | Test NLL | Test MSE | Test RMSE | Test MAE | Test MAPE |
|---------|---------|---------|---------|---------|---------|
| Concrete | 2.2148 ± 0.1384 | 15.2815 ± 4.5155 | 3.8640 ± 0.5923 | 2.2453 ± 0.2882 | 7.53% ± 1.08% |
| Energy | 0.1500 ± 0.1219 | 0.1829 ± 0.0555 | 0.4229 ± 0.0636 | 0.2856 ± 0.0337 | 1.20% ± 0.13% |
| Kin8nm | -1.2720 ± 0.0184 | 0.0050 ± 0.0002 | 0.0704 ± 0.0016 | 0.0549 ± 0.0014 | 10.51% ± 0.62% |
| Naval | -7.3279 ± 0.0186 | 0.0000 ± 0.0000 | 0.0001 ± 0.0000 | 0.0001 ± 0.0000 | 0.01% ± 0.00% |
| Power | 2.2778 ± 0.0219 | 9.0807 ± 1.4087 | 3.0046 ± 0.2306 | 2.0437 ± 0.0610 | 0.45% ± 0.01% |
| Protein | 1.7316 ± 0.0178 * | 12.4398 ± 0.1591 | 3.5269 ± 0.0226 | 2.2714 ± 0.0235 | 39.48% ± 1.08% |
| Wine | -2.7230 ± 0.1714 | 0.3927 ± 0.0421 | 0.6258 ± 0.0330 | 0.4903 ± 0.0252 | 8.93% ± 0.67% |
| Yacht | -0.7313 ± 0.2379 | 0.2015 ± 0.1627 | 0.4165 ± 0.1675 | 0.1873 ± 0.0633 | 4.20% ± 2.35% |

\* avg over 4 folds, as fold (3) give invalid NLL, so it's excluded from calculations

## OpenML-CTR23

below results are with pre-prcoessing to the dataset (most notebly one hot encoding)

Link inital results (video transcoding): https://www.kaggle.com/code/ayhamo/thesis-main?scriptVersionId=241417651

Link Part 1 (Grid -> health_insurance): https://www.kaggle.com/code/ayhamo/thesis-main?scriptVersionId=242935617

Link Part 2 (fifa -> space_ga): https://www.kaggle.com/code/ayhamo/thesis-main?scriptVersionId=243023781

| Dataset                           | Test NLL        | Test MSE                         | Test RMSE            | Test MAE               | Test MAPE                                       |
| :-------------------------------- | :------------------ | :----------------------------------- | :----------------------- | :------------------------- | :-------------------------------------------------- |
| grid_stability (361251)           | -4.1043 ± 0.0168    | 0.0000 ± 0.0000                      | 0.0055 ± 0.0003          | 0.0036 ± 0.0001            | 89.29% ± 126.63%                                    |
| video_transcoding (361252)        | 4.8826 ± 1.8184           | 12.9445 ± 2.0629                     | 3.5859 ± 0.2932          | 0.7569 ± 0.0444            | 5.28% ± 0.06%                                       |
| wave_energy (361253)              | 9.9437 ± 0.0121     | 20177032.2934 ± 4025032.8038          | 4470.6085 ± 436.6832     | 3020.2822 ± 81.7656        | 0.08% ± 0.00%                                       |
| sarcos (361254)                   | 1.9212 ± 0.0140     | 7.7924 ± 0.9086                      | 2.7871 ± 0.1564          | 1.6249 ± 0.0299            | 34.24% ± 4.15%          |
| california_housing (361255)       | 11.5564 ± 0.0366    | 1847555008.9483 ± 191653405.8820      | 42925.4313 ± 2227.6346   | 26436.8711 ± 630.3712       | 14.24% ± 0.58%                                      |
| cpu_activity (361256)             | 1.8632 ± 0.0360     | 7.1112 ± 0.6237                      | 2.6641 ± 0.1163          | 1.7302 ± 0.0804            | 2.82% ± 1.03%      |
| diamonds (361257)                 | 6.6750 ± 0.0251 *           | 281123.3248 ± 19605.7276            | 529.8782 ± 18.7715       | 264.1177 ± 8.4858         | 6.21% ± 0.14%                                       |
| kin8nm (361258)                   | -1.2821 ± 0.0268    | 0.0049 ± 0.0003                      | 0.0699 ± 0.0023          | 0.0545 ± 0.0021            | 10.42% ± 0.80%                                      |
| pumadyn32nh (361259)              | -2.4347 ± 0.0326    | 0.0004 ± 0.0000                      | 0.0209 ± 0.0007          | 0.0166 ± 0.0006            | 267.22% ± 71.73%                                   |
| miami_housing (361260)            | 11.8487 ± 0.0387    | 7570139372.3785 ± 618035857.1107     | 86935.5639 ± 3513.8431   | 39242.1456 ± 1296.9272      | 9.13% ± 0.38%                                       |
| cps88wages (361261)               | 6.9058 ± 0.0327 *           | 146638.2187 ± 34645.6972             | 380.4742 ± 43.3313       | 220.9764 ± 3.8062          | 50.91% ± 1.36%                                      |
| socmob (361264)                   | 2.0070 ± 0.1234     | 183.6488 ± 195.4986                   | 12.0529 ± 6.1948         | 4.2007 ± 0.9983            | 39.77% ± 3.43%       |
| kings_county (361266)             | 12.4506 ± 0.0278    | 14132590527.7067 ± 4757652673.5136   | 117462.1375 ± 18309.4725 | 60253.3985 ± 3399.2470      | 11.27% ± 0.25%                                      |
| brazilian_houses (361267)         | 8.3987 ± 0.0429 *          | 118880180.7745 ± 206370607.5488      | 7215.3342 ± 8174.2971    | 1540.2223 ± 278.9010       | 26.71% ± 0.85%                                      |
| fps_benchmark (361268)            | 1.7794 ± 0.0285    | 3.1837 ± 0.6340                     | 1.7752 ± 0.1797          | 0.9875 ± 0.0388            | 0.78% ± 0.03%                                    |
| health_insurance (361269)         | 1.9164 ± 0.0875 *           | 209.2058 ± 5.5444                    | 14.4627 ± 0.1922         | 11.1888 ± 0.2007           | 30.34% ± 1.11%     |
| fifa (361272)                     | 8.2709 ± 0.0526     | 108425961.4904 ± 23289032.7671       | 10351.0916 ± 1131.7528   | 3795.9989 ± 246.3346        | 82.09% ± 5.60%                                      |
| abalone (361234)                  | 0.2102 ± 0.0750     | 4.2753 ± 0.5141                      | 2.0641 ± 0.1218          | 1.4203 ± 0.0736            | 13.80% ± 0.64%                                      |
| airfoil_self_noise (361235)       | 1.0360 ± 0.0544     | 0.9484 ± 0.2265                      | 0.9666 ± 0.1188          | 0.6233 ± 0.0557            | 0.50% ± 0.04%                                       |
| auction_verification (361236)     | 6.1234 ± 0.1250     | 351583.2517 ± 86421.6278             | 588.2404 ± 74.5416       | 276.9586 ± 37.5523         | 4.98% ± 0.78%                                       |
| concrete_compressive_strength (361237) | 2.1725 ± 0.1066     | 14.9270 ± 6.7669                     | 3.7765 ± 0.8155          | 2.2283 ± 0.2765            | 7.32% ± 0.85%                                       |
| physiochemical_protein (361241)   | 1.7184 ± 0.0157 *           | 12.2795 ± 0.2773                     | 3.5040 ± 0.0396          | 2.2469 ± 0.0291            | 39.04% ± 1.57%        |
| superconductivity (361242)        | 2.7825 ± 0.0130 *           | 94.9896 ± 6.7674                     | 9.7401 ± 0.3472          | 5.4539 ± 0.1176            | 565.37% ± 611.25%                                   |
| geographical_origin_of_music (361243) | 2.2758 ± 0.2180     | 210.0773 ± 53.8257                   | 14.3860 ± 1.7668         | 9.7080 ± 1.1132            | 82.67% ± 18.09%                                     |
| solar_flare (361244)              | -3.8682 ± 0.2900 *           | 0.6290 ± 0.2475                      | 0.7794 ± 0.1468          | 0.3698 ± 0.0604            | 76.04% ± 5.83%       |
| naval_propulsion_plant (361247)   | -7.3417 ± 0.0137    | 0.0000 ± 0.0000                      | 0.0001 ± 0.0000          | 0.0001 ± 0.0000            | 0.01% ± 0.00%                                       |
| white_wine (361249)               | -2.4413 ± 0.2453 *           | 0.4841 ± 0.0471                      | 0.6949 ± 0.0348          | 0.5209 ± 0.0228            | 9.44% ± 0.54%                                       |
| red_wine (361250)                 | -2.6663 ± 0.1812    | 0.3901 ± 0.0467                      | 0.6234 ± 0.0383          | 0.4831 ± 0.0259            | 8.80% ± 0.74%                                       |
| Moneyball (361616)                | 4.4766 ± 0.0377     | 437.3483 ± 31.5674                   | 20.8995 ± 0.7481         | 16.4620 ± 0.7373           | 2.32% ± 0.10%                                       |
| energy_efficiency (361617)        | 0.1188 ± 0.0678     | 0.1692 ± 0.0479                      | 0.4073 ± 0.0577          | 0.2771 ± 0.0315            | 1.18% ± 0.13%                                       |
| forest_fires (361618)             | 2.1972 ± 0.3690     | 4063.1253 ± 7119.5515                | 46.2842 ± 43.8281        | 14.9761 ± 6.9952           | 281.92% ± 123.35%    |
| student_performance_por (361619)  | 0.6457 ± 0.2869     | 7.4567 ± 2.7828                      | 2.6828 ± 0.5091          | 1.9960 ± 0.2458            | 16.84% ± 5.02%     |
| QSAR_fish_toxicity (361621)       | 1.0693 ± 0.0887     | 0.7368 ± 0.0981                      | 0.8565 ± 0.0574          | 0.5947 ± 0.0388            | 23.73% ± 10.51%                                     |
| cars (361622)                     | 8.5861 ± 0.0723     | 4209088.5973 ± 649921.0897           | 2045.1781 ± 162.2813     | 1384.2685 ± 108.5005       | 6.39% ± 0.38%                                       |
| space_ga (361623)                 | -1.1043 ± 0.0664    | 0.0088 ± 0.0041                      | 0.0918 ± 0.0182          | 0.0647 ± 0.0042            | 12.75% ± 1.13%                                      |

Below datasets had some broken folds, in which NLL result was unstable, and as such was exceluced from from NLL avg:


|                         | Diamonds<br>(361257) | cps88wages<br>(361261) | brazilian_houses<br>(361267) | health_insurance<br>(361269) | physiochemical_protein<br>(361241) | superconductivity<br>(361242) | solar_flare<br>(361244) | white_wine<br>(361249) |
| :---------------------- | :------------------: | :--------------------: | :-------------------------: | :--------------------------: | :---------------------------------: | :--------------------------: | :---------------------: | :--------------------: |
| Number of Broken Folds        |          2           |           7            |              1              |               6              |                  2                  |               1              |            3            |           5            |


# XGBoost

Training metric was RMSE

Optuna optmization was done at: https://www.kaggle.com/code/ayhamo/thesis-main?scriptVersionId=242955599
for Concrete, Power and Protein, and generlized hyperparameters were picked then for all others

## UCI

Link: https://www.kaggle.com/code/ayhamo/thesis-main?scriptVersionId=243048628

| Dataset | Test NLL         | Test MSE         | Test RMSE        | Test MAE         | Test MAPE          |
|---|---|---|---|---|---|
| Concrete | 2.8815 ± 0.1419  | 19.3543 ± 5.1196 | 4.3590 ± 0.5944  | 2.7569 ± 0.2814  | 9.88% ± 1.24%      |
| Energy   | 0.2601 ± 0.2176  | 0.1085 ± 0.0490  | 0.3215 ± 0.0718  | 0.2103 ± 0.0317  | 0.99% ± 0.13%      |
| Kin8nm   | -0.6768 ± 0.0159 | 0.0151 ± 0.0005  | 0.1230 ± 0.0020  | 0.0960 ± 0.0018  | 19.38% ± 1.16%     |
| Naval    | -4.8440 ± 0.0347 | 0.0000 ± 0.0000  | 0.0019 ± 0.0001  | 0.0014 ± 0.0000  | 0.14% ± 0.00%      |
| Power    | 2.5384 ± 0.0740  | 9.4882 ± 1.4220  | 3.0718 ± 0.2290  | 2.1288 ± 0.0702  | 0.47% ± 0.02%      |
| Protein  | 2.6296 ± 0.0041  | 11.2619 ± 0.0928 | 3.3558 ± 0.0138  | 2.1932 ± 0.0163  | 42.91% ± 0.82%     |
| Wine     | 0.8435 ± 0.0794  | 0.3205 ± 0.0552  | 0.5642 ± 0.0466  | 0.3671 ± 0.0344  | 6.80% ± 0.79%      |
| Yacht    | 1.0351 ± 0.5998  | 0.8450 ± 0.8946  | 0.8036 ± 0.4464  | 0.3996 ± 0.1941  | 17.51% ± 19.08%    |

## OpenML-CTR23

link: https://www.kaggle.com/code/ayhamo/thesis-main?scriptVersionId=243073005

| Dataset                           | Test NLL         | Test MSE                              | Test RMSE             | Test MAE              | Test MAPE              |
|-----------------------------------|------------------|---------------------------------------|-----------------------|-----------------------|------------------------|
| grid_stability (361251)           | -3.3840 ± 0.0347 | 0.0001 ± 0.0000                       | 0.0082 ± 0.0003       | 0.0060 ± 0.0001       | 106.27% ± 98.86%       |
| video_transcoding (361252)        | 1.2074 ± 0.1804  | 0.7008 ± 0.2700                       | 0.8229 ± 0.1538       | 0.2617 ± 0.0105       | 3.46% ± 0.06%          |
| wave_energy (361253)              | 11.4008 ± 0.0131 | 468047280.7898 ± 12394958.3820        | 21632.5273 ± 284.6779 | 15653.6478 ± 212.7204 | 0.42% ± 0.01%          |
| sarcos (361254)                   | 2.2129 ± 0.0274  | 4.9011 ± 0.2683                       | 2.2130 ± 0.0607       | 1.2982 ± 0.0300       | 28.43% ± 7.25%         |
| california_housing (361255)       | 12.1221 ± 0.0439 | 1987546270.0629 ± 174126527.4015      | 44539.0520 ± 1954.2570| 28898.4819 ± 616.9129 | 16.19% ± 0.64%         |
| cpu_activity (361256)             | 2.2061 ± 0.0390  | 4.8418 ± 0.3768                       | 2.1987 ± 0.0857       | 1.5530 ± 0.0410       | 2.04% ± 0.13%          |
| diamonds (361257)                 | 7.7152 ± 0.0419  | 295368.5227 ± 24912.5720              | 542.9982 ± 22.8349    | 260.2848 ± 6.9082    | 6.10% ± 0.07%          |
| kin8nm (361258)                   | -0.6833 ± 0.0267 | 0.0150 ± 0.0008                       | 0.1222 ± 0.0032       | 0.0957 ± 0.0023       | 19.21% ± 1.39%         |
| pumadyn32nh (361259)              | -2.3997 ± 0.0311 | 0.0005 ± 0.0000                       | 0.0220 ± 0.0007       | 0.0175 ± 0.0006       | 264.96% ± 72.68%       |
| miami_housing (361260)            | 12.7733 ± 0.0422 | 7308453452.7606 ± 617317178.3352      | 85413.4382 ± 3605.2786| 40640.2614 ± 919.8348 | 9.90% ± 0.38%          |
| cps88wages (361261)               | 7.4444 ± 0.0966  | 174594.3211 ± 35460.1001              | 415.8071 ± 41.2163    | 249.6550 ± 5.0859    | 59.00% ± 1.98%         |
| socmob (361264)                   | 3.9970 ± 0.4094  | 240.1539 ± 206.5902                   | 14.3132 ± 5.9402      | 5.4653 ± 1.4149       | 57.12% ± 6.11%         |
| kings_county (361266)             | 13.0982 ± 0.1434 | 14563350552.6073 ± 4486980316.7839    | 119354.3262 ± 17829.6204| 61819.0575 ± 2373.5084 | 11.71% ± 0.36%         |
| brazilian_houses (361267)         | 9.7582 ± 0.4411  | 27117261.1990 ± 28250139.7710         | 4653.9023 ± 2336.3335 | 1629.2150 ± 106.1944  | 30.40% ± 1.07%         |
| fps_benchmark (361268)            | 1.5679 ± 0.1724  | 1.4339 ± 0.5376                       | 1.1784 ± 0.2124       | 0.5659 ± 0.0315       | 0.47% ± 0.02%          |
| health_insurance (361269)         | 4.2293 ± 0.0172  | 276.2371 ± 9.3639                     | 16.6180 ± 0.2839      | 12.3405 ± 0.2230      | 36.96% ± 1.12%         |
| fifa (361272)                     | 10.5315 ± 0.1003 | 83761065.2055 ± 14617882.5601         | 9112.4963 ± 850.5738  | 3824.3456 ± 220.4890  | 92.33% ± 4.74%         |
| abalone (361234)                  | 2.2153 ± 0.0436  | 4.9362 ± 0.4456                       | 2.2196 ± 0.0985       | 1.5798 ± 0.0562       | 15.78% ± 0.48%         |
| airfoil_self_noise (361235)       | 1.6500 ± 0.1370  | 1.6446 ± 0.4137                       | 1.2715 ± 0.1672       | 0.8358 ± 0.0887       | 0.68% ± 0.07%          |
| auction_verification (361236)     | 7.6075 ± 0.1977  | 256318.9940 ± 100614.9921             | 496.7053 ± 97.9943    | 224.6542 ± 38.0101    | 15.97% ± 4.96%         |
| concrete_compressive_strength (361237)| 2.8521 ± 0.1681  | 18.5999 ± 6.4485                     | 4.2506 ± 0.7296       | 2.7302 ± 0.2643       | 9.73% ± 1.12%          |
| physiochemical_protein (361241)   | 2.6239 ± 0.0093  | 11.1340 ± 0.2087                      | 3.3366 ± 0.0312       | 2.1821 ± 0.0234       | 43.31% ± 1.12%         |
| superconductivity (361242)         | 3.6412 ± 0.0325  | 85.3380 ± 5.5108                      | 9.2330 ± 0.2990       | 4.9466 ± 0.1418       | 509.47% ± 612.51%      |
| geographical_origin_of_music (361243)| 4.1555 ± 0.0972  | 242.9032 ± 51.3564                   | 15.5055 ± 1.5756      | 11.6030 ± 0.9604      | 99.70% ± 26.05%        |
| solar_flare (361244)              | 1.3490 ± 0.0979  | 0.8863 ± 0.1757                       | 0.9369 ± 0.0922       | 0.4629 ± 0.0543       | 81.09% ± 14.63%        |
| naval_propulsion_plant (361247)   | -4.8536 ± 0.0395 | 0.0000 ± 0.0000                       | 0.0019 ± 0.0001       | 0.0014 ± 0.0000       | 0.14% ± 0.00%          |
| white_wine (361249)               | 0.8752 ± 0.0444  | 0.3384 ± 0.0298                       | 0.5811 ± 0.0257       | 0.3663 ± 0.0128       | 6.58% ± 0.34%          |
| red_wine (361250)                 | 0.8218 ± 0.0817  | 0.3067 ± 0.0454                       | 0.5521 ± 0.0429       | 0.3594 ± 0.0304       | 6.69% ± 0.73%          |
| Moneyball (361616)                | 4.5787 ± 0.0389  | 556.7851 ± 42.9726                    | 23.5786 ± 0.9133      | 18.6725 ± 0.7344      | 2.65% ± 0.11%          |
| energy_efficiency (361617)        | 0.1469 ± 0.2249  | 0.0873 ± 0.0430                       | 0.2876 ± 0.0674       | 0.1929 ± 0.0293       | 0.93% ± 0.15%          |
| forest_fires (361618)             | 5.3570 ± 0.5472  | 4983.3606 ± 6490.6476                 | 60.0951 ± 37.0398     | 22.8329 ± 7.0027      | 587.35% ± 290.72%      |
| student_performance_por (361619)  | 2.4024 ± 0.1836  | 7.6262 ± 2.6984                       | 2.7174 ± 0.4920       | 2.0127 ± 0.2579       | 16.60% ± 4.51%         |
| QSAR_fish_toxicity (361621)       | 1.2945 ± 0.0708  | 0.7874 ± 0.1119                       | 0.8851 ± 0.0628       | 0.6240 ± 0.0401       | 23.39% ± 7.92%         |
| cars (361622)                     | 9.1891 ± 0.1112  | 5747673.8650 ± 1236742.8127           | 2383.1968 ± 260.8578  | 1613.3961 ± 178.2680  | 7.59% ± 0.70%          |
| space_ga (361623)                 | -0.8113 ± 0.1587 | 0.0123 ± 0.0050                       | 0.1090 ± 0.0197       | 0.0771 ± 0.0058       | 15.16% ± 1.36%         |

# CatBoost

Training metric was RMSE

hyperparameters ranges were obtained from catboost paper,
general paramters are taken from: https://github.com/catboost/benchmarks/blob/master/quality_benchmarks/comparison_description.pdf

as they say, out-of-the-box performance, so below are generally used, not optimized per dataset

## UCI

Link: https://www.kaggle.com/code/ayhamo/thesis-main?scriptVersionId=243499965

| Dataset     | Test NLL         | Test MSE         | Test RMSE        | Test MAE         | Test MAPE        |
|-------------------------|--------------|--------------|--------------|--------------|---------------|
| Concrete  | 2.9509 ± 0.1093  | 21.9038 ± 4.6034 | 4.6536 ± 0.4982  | 3.2815 ± 0.2722  | 11.18% ± 1.06%     |
| Energy    | 0.7114 ± 0.1281  | 0.2511 ± 0.0670  | 0.4969 ± 0.0651  | 0.3581 ± 0.0392  | 1.65% ± 0.19%      |
| Kin8nm    | -0.8429 ± 0.0278 | 0.0109 ± 0.0006  | 0.1042 ± 0.0029  | 0.0814 ± 0.0023  | 16.42% ± 0.99%     |
| Naval     | -4.8125 ± 0.0387 | 0.0000 ± 0.0000  | 0.0020 ± 0.0001  | 0.0015 ± 0.0001  | 0.16% ± 0.01%      |
| Power     | 2.7122 ± 0.0476  | 13.3446 ± 1.2729 | 3.6489 ± 0.1740  | 2.7322 ± 0.0728  | 0.60% ± 0.02%      |
| Protein   | 2.8423 ± 0.0064  | 17.2313 ± 0.2199 | 4.1510 ± 0.0265  | 3.1583 ± 0.0243  | 72.94% ± 1.24%     |
| Wine      | 0.9058 ± 0.0677  | 0.3617 ± 0.0518  | 0.6000 ± 0.0417  | 0.4569 ± 0.0280  | 8.36% ± 0.67%      |
| Yacht     | 1.3185 ± 0.7344  | 2.6731 ± 4.7011  | 1.2210 ± 1.0873  | 0.5489 ± 0.2980  | 28.50% ± 19.43%    |

## OpenML-CTR23

Link: https://www.kaggle.com/code/ayhamo/thesis-main?scriptVersionId=243501583

| Dataset                                | Test NLL         | Test MSE                          | Test RMSE                 | Test MAE                  | Test MAPE             |
|----------------------------------------|------------------|-----------------------------------|---------------------------|---------------------------|-----------------------|
| grid_stability (361251)                | -3.3940 ± 0.0289 | 0.0001 ± 0.0000                   | 0.0081 ± 0.0002           | 0.0060 ± 0.0002           | 82.50% ± 47.64%       |
| video_transcoding (361252)             | 2.1996 ± 0.0568  | 4.7962 ± 0.5461                   | 2.1865 ± 0.1244           | 0.9976 ± 0.0294           | 14.21% ± 0.27%        |
| wave_energy (361253)                   | 10.4935 ± 0.0098 | 76239808.9842 ± 1503636.3610     | 8731.1192 ± 85.8271       | 6730.9778 ± 80.8589       | 0.18% ± 0.00%         |
| sarcos (361254)                        | 2.7334 ± 0.0278  | 13.8816 ± 0.7761                  | 3.7244 ± 0.1037           | 2.6317 ± 0.0669           | 56.42% ± 9.47%        |
| california_housing (361255)            | 12.2049 ± 0.0382 | 2343352218.6663 ± 181510180.0063 | 48372.4446 ± 1859.7897     | 32656.5028 ± 572.7495     | 18.44% ± 0.76%        |
| cpu_activity (361256)                  | 2.3874 ± 0.1377  | 7.2375 ± 2.3588                   | 2.6604 ± 0.3996           | 1.7882 ± 0.0516           | 2.44% ± 0.32%         |
| diamonds (361257)                      | 8.2245 ± 0.0199  | 815862.1613 ± 32506.8275         | 903.0721 ± 17.9692        | 486.7339 ± 8.7350         | 14.20% ± 0.32%        |
| kin8nm (361258)                        | -0.8593 ± 0.0249 | 0.0105 ± 0.0005                   | 0.1025 ± 0.0025           | 0.0794 ± 0.0019           | 16.04% ± 1.13%        |
| pumadyn32nh (361259)                   | -2.4390 ± 0.0298 | 0.0004 ± 0.0000                   | 0.0211 ± 0.0006           | 0.0168 ± 0.0006           | 258.20% ± 63.25%      |
| miami_housing (361260)                 | 12.8148 ± 0.0439 | 7943239375.9150 ± 694338414.4499 | 89039.4681 ± 3900.3201     | 47673.4748 ± 1125.9585    | 12.28% ± 0.49%        |
| cps88wages (361261)                    | 7.3536 ± 0.1096  | 146472.7367 ± 34682.6379         | 380.2550 ± 43.3457        | 224.5756 ± 3.6040         | 54.01% ± 1.56%        |
| socmob (361264)                        | 4.4516 ± 0.3219  | 534.6711 ± 377.6231               | 21.8913 ± 7.4460          | 8.7804 ± 1.5335           | 94.56% ± 20.73%       |
| kings_county (361266)                  | 13.2584 ± 0.1154 | 19785103189.0316 ± 5285839068.8266| 139586.5151 ± 17340.9336  | 80834.4082 ± 3718.5989   | 15.50% ± 0.29%        |
| brazilian_houses (361267)              | 10.2290 ± 0.7410 | 126339454.3863 ± 187933986.3218  | 8861.3580 ± 6914.8962      | 2022.2317 ± 340.1070      | 38.43% ± 3.56%        |
| fps_benchmark (361268)                 | 2.5643 ± 0.0662  | 9.9276 ± 1.3137                   | 3.1439 ± 0.2079           | 2.3720 ± 0.1367           | 2.09% ± 0.12%         |
| health_insurance (361269)              | 4.0930 ± 0.0147  | 210.2858 ± 6.1212                 | 14.4997 ± 0.2124          | 11.2185 ± 0.2118          | 30.99% ± 0.82%        |
| fifa (361272)                          | 10.5445 ± 0.0945 | 85819685.8628 ± 14559520.1413    | 9227.0417 ± 825.4616      | 4036.6813 ± 209.7574      | 109.60% ± 6.14%       |
| abalone (361234)                       | 2.2790 ± 0.0494  | 5.6123 ± 0.5619                   | 2.3661 ± 0.1176           | 1.6822 ± 0.0698           | 16.65% ± 0.59%        |
| airfoil_self_noise (361235)            | 2.0816 ± 0.0996  | 3.8381 ± 0.7613                   | 1.9495 ± 0.1939           | 1.4429 ± 0.1265           | 1.15% ± 0.10%         |
| auction_verification (361236)          | 8.4366 ± 0.1059  | 1273506.0958 ± 264753.6442       | 1122.3291 ± 117.8285       | 582.0016 ± 63.5674        | 33.19% ± 8.81%        |
| concrete_compressive_strength (361237) | 2.9263 ± 0.1499  | 21.3442 ± 6.7796                  | 4.5657 ± 0.7065           | 3.2067 ± 0.3618           | 10.76% ± 1.49%        |
| physiochemical_protein (361241)        | 2.8288 ± 0.0083  | 16.7753 ± 0.2781                  | 4.0956 ± 0.0339           | 3.1104 ± 0.0263           | 72.33% ± 1.60%        |
| superconductivity (361242)             | 3.8052 ± 0.0260  | 118.3788 ± 6.1156                 | 10.8765 ± 0.2820          | 7.0011 ± 0.1593           | 852.45% ± 683.46%     |
| geographical_origin_of_music (361243)  | 4.1441 ± 0.1004  | 237.6736 ± 50.8530                | 15.3343 ± 1.5916          | 11.6083 ± 0.9589          | 105.72% ± 31.05%      |
| solar_flare (361244)                   | 1.1795 ± 0.1585  | 0.6533 ± 0.2302                   | 0.7973 ± 0.1330           | 0.4269 ± 0.0499           | 68.68% ± 5.51%        |
| naval_propulsion_plant (361247)        | -4.8145 ± 0.0296 | 0.0000 ± 0.0000                   | 0.0020 ± 0.0001           | 0.0015 ± 0.0000           | 0.16% ± 0.00%         |
| white_wine (361249)                    | 1.0122 ± 0.0336  | 0.4443 ± 0.0291                   | 0.6662 ± 0.0221           | 0.5171 ± 0.0124           | 9.11% ± 0.33%         |
| red_wine (361250)                      | 0.8967 ± 0.0574  | 0.3542 ± 0.0412                   | 0.5942 ± 0.0344           | 0.4531 ± 0.0288           | 8.32% ± 0.67%         |
| Moneyball (361616)                     | 4.8429 ± 0.0434  | 945.3987 ± 78.6804                | 30.7196 ± 1.3050          | 24.5330 ± 1.2009          | 3.47% ± 0.17%         |
| energy_efficiency (361617)             | 0.6657 ± 0.1477  | 0.2313 ± 0.0661                   | 0.4759 ± 0.0691           | 0.3409 ± 0.0427           | 1.57% ± 0.20%         |
| forest_fires (361618)                  | 5.1207 ± 0.6958  | 4224.7676 ± 6231.5007             | 51.9375 ± 39.0803         | 20.5135 ± 8.1042          | 444.14% ± 189.38%     |
| student_performance_por (361619)       | 2.3829 ± 0.1677  | 7.2460 ± 2.2857                   | 2.6569 ± 0.4321           | 2.0082 ± 0.2343           | 16.81% ± 4.07%        |
| QSAR_fish_toxicity (361621)            | 1.3016 ± 0.0728  | 0.7991 ± 0.1163                   | 0.8915 ± 0.0649           | 0.6331 ± 0.0434           | 24.95% ± 9.11%        |
| cars (361622)                          | 9.0754 ± 0.0928  | 4543009.6012 ± 776057.9047       | 2123.0642 ± 188.7010      | 1520.1781 ± 118.4068     | 7.30% ± 0.54%         |
| space_ga (361623)                      | -0.7875 ± 0.1451 | 0.0127 ± 0.0046                   | 0.1113 ± 0.0181           | 0.0802 ± 0.0055           | 16.04% ± 1.49%        |


# TabResNet

Training metric was RMSE

hyperparameters were taken from the library testing on bunch of datasets: https://jrzaurin.github.io/infinitoml/2021/05/28/pytorch-widedeep_iv.html

## UCI

Link: https://www.kaggle.com/code/ayhamo/thesis-main?scriptVersionId=245279742

| Dataset                      | Test NLL             | Test MSE              | Test RMSE            | Test MAE             | Test MAPE            |
|------------------------------|----------------------|-----------------------|----------------------|----------------------|----------------------|
| Concrete                     | 3.1696 ± 0.0763      | 33.4399 ± 5.1269      | 5.7658 ± 0.4423      | 4.2128 ± 0.2875      | 14.07% ± 1.34%       |
| Energy                       | 3.6134 ± 1.3055      | 33.0509 ± 53.7260     | 4.7157 ± 3.2883      | 4.1615 ± 3.0067      | 20.56% ± 14.06%      |
| Kin8nm                       | -0.8595 ± 0.1725     | 0.0098 ± 0.0026       | 0.0982 ± 0.0125      | 0.0800 ± 0.0114      | 12.73% ± 1.37%       |
| Naval       | -2.5209 ± 0.6835     | 0.0002 ± 0.0002       | 0.0146 ± 0.0051      | 0.0122 ± 0.0046      | 1.24% ± 0.46%        |
| Power                  | 4.3301 ± 1.5719      | 106.1714 ± 90.4414    | 9.5650 ± 3.8317      | 8.2903 ± 3.9516      | 1.82% ± 0.87%        |
| Protein   | 3.4960 ± 0.3256      | 58.5130 ± 28.5243     | 7.4585 ± 1.6983      | 6.1844 ± 1.4931      | 215.60% ± 89.02%     |
| Wine             | 1.0447 ± 0.0536      | 0.4706 ± 0.0462       | 0.6852 ± 0.0342      | 0.5309 ± 0.0243      | 9.40% ± 0.55%        |
| Yacht                        | 2.5100 ± 0.4910      | 12.0495 ± 8.3310      | 3.2256 ± 1.2827      | 1.7955 ± 0.7296      | 53.99% ± 26.87%      |

## OpenML-CTR23

Link: https://www.kaggle.com/code/ayhamo/thesis-main?scriptVersionId=245280269

| Dataset                               | Test NLL              | Test MSE                                | Test RMSE                 | Test MAE                  | Test MAPE              |
|---------------------------------------|-----------------------|-----------------------------------------|---------------------------|---------------------------|------------------------|
| grid_stability (361251)               | -3.3527 ± 0.1207      | 0.0001 ± 0.0000                         | 0.0085 ± 0.0010           | 0.0067 ± 0.0010           | 109.22% ± 97.97%       |
| video_transcoding (361252)            | 1.8625 ± 0.1307       | 2.5143 ± 0.7170                         | 1.5712 ± 0.2139           | 0.7278 ± 0.0839           | 14.99% ± 2.43%         |
| wave_energy (361253)                  | 13.2414 ± 2.4484      | 630536198.4836 ± 367691990.0705         | 23975.3117 ± 7464.6250    | 21608.9752 ± 7747.6771    | 0.57% ± 0.21%          |
| sarcos (361254)                       | 2.6505 ± 0.0916       | 11.7916 ± 2.0053                        | 3.4211 ± 0.2958           | 2.5162 ± 0.2529           | 44.16% ± 7.36%         |
| california_housing (361255)           | 12.6498 ± 0.2177      | 5731829172.8496 ± 2219063887.9075       | 74382.9958 ± 14106.7043   | 54281.4934 ± 11386.9203   | 28.80% ± 5.90%         |
| cpu_activity (361256)                 | 3.3018 ± 1.0781       | 39.2216 ± 58.7860                       | 5.2785 ± 3.3704           | 4.4482 ± 3.2813           | 5.46% ± 3.84%          |
| diamonds (361257)                     | 7.9975 ± 0.0529       | 520249.2515 ± 56704.3436                | 720.2492 ± 38.6047        | 354.8036 ± 19.2909        | 9.13% ± 0.18%          |
| kin8nm (361258)                       | -1.0459 ± 0.1522      | 0.0072 ± 0.0021                         | 0.0843 ± 0.0114           | 0.0674 ± 0.0103           | 11.48% ± 1.24%         |
| pumadyn32nh (361259)                  | -2.1367 ± 0.1901      | 0.0009 ± 0.0003                         | 0.0291 ± 0.0053           | 0.0231 ± 0.0041           | 235.32% ± 46.85%       |
| miami_housing (361260)                | 13.0223 ± 0.0775      | 12115593482.2717 ± 1902481405.5462      | 109740.0256 ± 8527.6178   | 58181.0137 ± 7738.8015   | 15.09% ± 2.80%         |
| cps88wages (361261)                   | 7.4074 ± 0.0983       | 162233.0056 ± 33916.8092                | 400.7276 ± 40.6246        | 232.3680 ± 4.1957         | 51.09% ± 1.86%         |
| socmob (361264)                       | 4.7709 ± 0.3154       | 983.5829 ± 600.8121                     | 29.9612 ± 9.2687          | 11.8413 ± 2.9483          | 99.51% ± 8.73%         |
| kings_county (361266)                 | 13.5204 ± 0.1471      | 33959113254.2392 ± 10880851510.4136     | 182138.1590 ± 28014.3590  | 107033.7765 ± 4868.8705   | 20.15% ± 0.35%         |
| brazilian_houses (361267)             | 10.2135 ± 0.7884      | 184088113.1992 ± 360229625.4841        | 9514.5037 ± 9672.7624     | 2191.9918 ± 385.5978      | 37.13% ± 3.27%         |
| fps_benchmark (361268)                | 3.5078 ± 0.7474       | 50.2346 ± 43.5918                       | 6.4106 ± 3.0230           | 5.2784 ± 2.6356           | 4.13% ± 1.93%          |
| health_insurance (361269)             | 4.3208 ± 0.0156       | 331.6207 ± 10.4357                      | 18.2082 ± 0.2850          | 12.7059 ± 0.3059          | 41.91% ± 1.30%         |
| fifa (361272)                         | 10.8324 ± 0.0893      | 152383782.9666 ± 24970103.3691          | 12299.6116 ± 1050.3992   | 4830.2042 ± 222.0408      | 94.73% ± 5.14%         |
| abalone (361234)                      | 2.5464 ± 0.1344       | 9.6863 ± 2.7162                         | 3.0871 ± 0.3949           | 2.2393 ± 0.2997           | 22.19% ± 2.34%         |
| airfoil_self_noise (361235)           | 3.2631 ± 0.0975       | 40.4833 ± 8.2148                        | 6.3318 ± 0.6253           | 5.1157 ± 0.5526           | 4.11% ± 0.48%          |
| auction_verification (361236)         | 8.1466 ± 0.0978       | 710135.0436 ± 140923.8559               | 838.6441 ± 82.5291        | 505.6831 ± 51.5283        | 41.00% ± 12.77%        |
| concrete_compressive_strength (361237)| 3.1493 ± 0.0926       | 32.1677 ± 6.8079                        | 5.6442 ± 0.5576           | 4.2434 ± 0.3239           | 14.32% ± 1.21%         |
| physiochemical_protein (361241)       | 3.3559 ± 0.2899       | 52.2450 ± 30.8531                       | 6.9638 ± 1.9365           | 5.4212 ± 1.6191           | 161.26% ± 93.90%       |
| superconductivity (361242)            | 4.1125 ± 0.0784       | 221.0088 ± 35.7952                      | 14.8198 ± 1.1751          | 9.2058 ± 0.5307           | 825.12% ± 510.22%      |
| geographical_origin_of_music (361243) | 4.2619 ± 0.1136       | 301.4084 ± 62.1681                      | 17.2594 ± 1.8767          | 12.3249 ± 1.2259          | 90.42% ± 23.67%        |
| solar_flare (361244)                  | 1.3139 ± 0.1474       | 0.8453 ± 0.2414                         | 0.9099 ± 0.1320           | 0.4385 ± 0.0619           | 84.23% ± 10.62%        |
| naval_propulsion_plant (361247)       | -2.7499 ± 0.6620      | 0.0002 ± 0.0001                         | 0.0129 ± 0.0051           | 0.0106 ± 0.0043           | 1.09% ± 0.43%          |
| white_wine (361249)                   | 1.1139 ± 0.0425       | 0.5438 ± 0.0442                         | 0.7368 ± 0.0302           | 0.5720 ± 0.0178           | 9.89% ± 0.39%          |
| red_wine (361250)                     | 1.0233 ± 0.0858       | 0.4552 ± 0.0694                         | 0.6726 ± 0.0529           | 0.5158 ± 0.0519           | 9.19% ± 0.89%          |
| Moneyball (361616)                    | 5.8670 ± 0.0916       | 7313.9748 ± 1285.6976                   | 85.1957 ± 7.4612          | 66.6351 ± 7.0045          | 9.26% ± 0.94%          |
| energy_efficiency (361617)            | 3.3493 ± 1.4883       | 13.6008 ± 9.9648                        | 3.4018 ± 1.4244           | 2.8877 ± 1.3656           | 14.98% ± 6.24%         |
| forest_fires (361618)                 | 5.5098 ± 0.4610       | 5588.7016 ± 5658.2627                   | 66.7953 ± 33.5722         | 24.8320 ± 10.6065         | 647.29% ± 509.56%      |
| student_performance_por (361619)      | 2.6156 ± 0.1127       | 11.0542 ± 2.3991                        | 3.3047 ± 0.3645           | 2.5507 ± 0.3290           | 20.99% ± 3.63%         |
| QSAR_fish_toxicity (361621)           | 1.3620 ± 0.0850       | 0.9016 ± 0.1593                         | 0.9460 ± 0.0817           | 0.6768 ± 0.0347           | 23.42% ± 7.69%         |
| cars (361622)                         | 12.1178 ± 1.8513      | 604268570.6450 ± 748883264.5074         | 20568.5844 ± 13461.1258   | 18323.8385 ± 13103.9223   | 101.36% ± 80.40%       |
| space_ga (361623)                     | -0.5381 ± 0.1230      | 0.0207 ± 0.0065                         | 0.1423 ± 0.0199           | 0.1012 ± 0.0045           | 20.44% ± 2.01%         |
