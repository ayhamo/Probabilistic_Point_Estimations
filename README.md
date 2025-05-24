# Will be filled at later date, Currently will host results

The runs are found in https://www.kaggle.com/code/ayhamo/thesis-main in Version tab

# TabResFlow

## UCI
20 folds for each dataset, but protien is 5 folds!

Link: https://www.kaggle.com/code/ayhamo/thesis-main/log?scriptVersionId=240458028

| Dataset | Test NLL | Test MSE | Test RMSE | Test MAE | Test MAPE | Kiran TabResFlow NLL | Kiran TabResFlow RMSE |
|---------|---------|---------|---------|---------|---------|---------|---------|
| Concrete | 2.8915 ± 0.1198 | 27.4820 ± 5.8639 | 5.2117 ± 0.5660 | 3.4602 ± 0.2921 | 11.31% ± 1.25% | 2.90 ± 0.15 | 5.01 ± 0.70 |
| Energy | 0.6444 ± 0.1246 | 0.4434 ± 0.2697 | 0.6453 ± 0.1645 | 0.4254 ± 0.0695 | 1.95% ± 0.27% | 0.77 ± 0.19 | 1.45 ± 2.24 |
| Kin8nm | -1.2842 ± 0.0320 | 0.0053 ± 0.0004 | 0.0729 ± 0.0030 | 0.0560 ± 0.0019 | 10.36% ± 0.63% | -1.29 ± 0.04 | 0.07 ± 0.00 |
| Naval | -5.3268 ± 0.1114 | 0.0000 ± 0.0000 | 0.0015 ± 0.0004 | 0.0010 ± 0.0001 | 0.10% ± 0.01% | -5.30 ± 0.11 | 0.00 ± 0.00 |
| Power | 2.5808 ± 0.0220 | 16.2194 ± 1.4487 | 4.0233 ± 0.1809 | 2.8301 ± 0.0863 | 0.62% ± 0.02% | 2.60 ± 0.04 | 3.98 ± 0.19 |
| Protein | 1.9293 ± 0.0184 | 19.0796 ± 0.5625 | 4.3675 ± 0.0643 | 2.5278 ± 0.0427 | 11.06% ± 1.43% | 1.95 ± 0.04 | 4.44 ± 0.10 |
| Wine | -0.5317 ± 0.2270 | 0.5053 ± 0.0745 | 0.7089 ± 0.0524 | 0.4446 ± 0.0469 | 8.02% ± 1.02% | -0.85 ± 0.27 | 0.39 ± 0.06 |
| Yacht | 0.5558 ± 0.2196 | 2.7534 ± 3.8523 | 1.4134 ± 0.8693 | 0.6382 ± 0.2814 | 30.19% ± 25.13% | 0.67 ± 0.32 | 0.47 ± 0.11 |

## OpenML-CTR23
Later due to need for hyperparamter tuning

# TabPFN

## UCI

20 folds for each dataset, but protien is 5 folds!

Link: https://www.kaggle.com/code/ayhamo/thesis-main?scriptVersionId=241417558

# Summary of Dataset Evaluations

| Dataset | Test NLL | Test MSE | Test RMSE | Test MAE | Test MAPE |
|---------|---------|---------|---------|---------|---------|
| Concrete | 2.2148 ± 0.1384 | 15.2815 ± 4.5155 | 3.8640 ± 0.5923 | 2.2453 ± 0.2882 | 7.53% ± 1.08% |
| Energy | 0.1500 ± 0.1219 | 0.1829 ± 0.0555 | 0.4229 ± 0.0636 | 0.2856 ± 0.0337 | 1.20% ± 0.13% |
| Kin8nm | -1.2720 ± 0.0184 | 0.0050 ± 0.0002 | 0.0704 ± 0.0016 | 0.0549 ± 0.0014 | 10.51% ± 0.62% |
| Naval | -7.3279 ± 0.0186 | 0.0000 ± 0.0000 | 0.0001 ± 0.0000 | 0.0001 ± 0.0000 | 0.01% ± 0.00% |
| Power | 2.2778 ± 0.0219 | 9.0807 ± 1.4087 | 3.0046 ± 0.2306 | 2.0437 ± 0.0610 | 0.45% ± 0.01% |
| Protein | inf ± nan | 12.4398 ± 0.1591 | 3.5269 ± 0.0226 | 2.2714 ± 0.0235 | 14404776413472722.00% ± 983114151567592.25% |
| Wine | -2.7230 ± 0.1714 | 0.3927 ± 0.0421 | 0.6258 ± 0.0330 | 0.4903 ± 0.0252 | 8.93% ± 0.67% |
| Yacht | -0.7313 ± 0.2379 | 0.2015 ± 0.1627 | 0.4165 ± 0.1675 | 0.1873 ± 0.0633 | 4.20% ± 2.35% |



## OpenML-CTR23 (No Pre-prcoess)

10 Folds for each dataset, below is without pre-prcoessing to the dataset

Link: https://www.kaggle.com/code/ayhamo/thesis-main?scriptVersionId=240691543


| Dataset                           | Avg Test NLL        | Avg Test MSE                         | Avg Test RMSE            | Avg Test MAE               | Avg Test MAPE                                       |
| :-------------------------------- | :------------------ | :----------------------------------- | :----------------------- | :------------------------- | :-------------------------------------------------- |
| grid_stability (361251)           | -4.1043 ± 0.0168    | 0.0000 ± 0.0000                      | 0.0055 ± 0.0003          | 0.0036 ± 0.0001            | 95.38% ± 125.49%                                    |
| video_transcoding (361252)        | inf ± nan           | 12.1467 ± 1.7606                     | 3.4756 ± 0.2592          | 0.7509 ± 0.0428            | 5.36% ± 0.09%                                       |
| wave_energy (361253)              | 9.9437 ± 0.0121     | 20177032.2934 ± 4025032.8038         | 4470.6085 ± 436.6832      | 3020.2822 ± 81.7656        | 0.08% ± 0.00%                                       |
| sarcos (361254)                   | 1.9212 ± 0.0140     | 7.7924 ± 0.9086                      | 2.7871 ± 0.1564          | 1.6249 ± 0.0299            | 993457449473995.38% ± 429951039506990.75%          |
| california_housing (361255)       | 11.5564 ± 0.0366    | 1847555008.9483 ± 191653405.8820     | 42925.4313 ± 2227.6346    | 26436.8711 ± 630.3712       | 14.24% ± 0.58%                                      |
| cpu_activity (361256)             | 1.8632 ± 0.0360     | 7.1112 ± 0.6237                      | 2.6641 ± 0.1163          | 1.7302 ± 0.0804            | 145069519335287776.00% ± 30210526094306520.00%     |
| diamonds (361257)                 | inf ± nan           | 281937.4718 ± 18052.4314             | 530.6950 ± 17.3277        | 263.4526 ± 7.4544          | 6.13% ± 0.13%                                       |
| kin8nm (361258)                   | -1.2821 ± 0.0268    | 0.0049 ± 0.0003                      | 0.0699 ± 0.0023          | 0.0545 ± 0.0021            | 10.42% ± 0.80%                                      |
| pumadyn32nh (361259)              | -2.4347 ± 0.0326    | 0.0004 ± 0.0000                      | 0.0209 ± 0.0007          | 0.0166 ± 0.0006            | 341.77% ± 214.82%                                   |
| miami_housing (361260)            | 11.8487 ± 0.0387    | 7570139372.3785 ± 618035857.1107     | 86935.5639 ± 3513.8431    | 39242.1456 ± 1296.9272      | 9.13% ± 0.38%                                       |
| cps88wages (361261)               | inf ± nan           | 146601.8929 ± 34555.1383             | 380.4383 ± 43.2272        | 220.8898 ± 3.7169          | 50.92% ± 1.38%                                      |
| socmob (361264)                   | 2.0527 ± 0.1083     | 203.8422 ± 213.1755                  | 12.7231 ± 6.4780         | 4.3438 ± 1.1214            | 43658596868708648.00% ± 10811044425219140.00%      |
| kings_county (361266)             | 12.4317 ± 0.0288    | 14444533551.6150 ± 5462801080.7671   | 118395.1973 ± 20666.6596  | 59474.1276 ± 3571.9831      | 11.08% ± 0.26%                                      |
| brazilian_houses (361267)         | inf ± nan           | 108666803.1819 ± 183454378.3235     | 7116.7821 ± 7616.9689     | 1533.1313 ± 266.7270       | 26.42% ± 0.84%                                      |
| health_insurance (361269)         | inf ± nan           | 209.4210 ± 5.9440                    | 14.4699 ± 0.2060         | 11.1669 ± 0.2105           | 2081349599613515264.00% ± 54831739662726896.00%    |
| fifa (361272)                     | 8.2750 ± 0.0483     | 109512603.7467 ± 22378569.8145       | 10407.4792 ± 1094.0663    | 3842.0044 ± 241.3553       | 84.77% ± 5.77%                                      |
| abalone (361234)                  | 0.2096 ± 0.0731     | 4.2660 ± 0.5036                      | 2.0620 ± 0.1196          | 1.4207 ± 0.0726            | 13.81% ± 0.64%                                      |
| airfoil_self_noise (361235)       | 1.0360 ± 0.0544     | 0.9484 ± 0.2265                      | 0.9666 ± 0.1188          | 0.6233 ± 0.0557            | 0.50% ± 0.04%                                       |
| auction_verification (361236)     | 6.1594 ± 0.1232     | 1103101.5308 ± 385257.1967           | 1035.0412 ± 178.3009     | 415.1999 ± 75.9336        | 30.55% ± 11.73%                                     |
| concrete_compressive_strength (361237) | 2.1725 ± 0.1066     | 14.9270 ± 6.7669                     | 3.7765 ± 0.8155          | 2.2283 ± 0.2765            | 7.32% ± 0.85%                                       |
| physiochemical_protein (361241)   | inf ± nan           | 12.2795 ± 0.2773                     | 3.5040 ± 0.0396          | 2.2469 ± 0.0291            | 15055142516309920.00% ± 3342818623822434.00%       |
| superconductivity (361242)        | inf ± nan           | 94.9896 ± 6.7674                     | 9.7401 ± 0.3472          | 5.4539 ± 0.1176            | 565.37% ± 611.25%                                   |
| geographical_origin_of_music (361243) | 2.2758 ± 0.2180     | 210.0773 ± 53.8257                   | 14.3860 ± 1.7668         | 9.7080 ± 1.1132            | 82.67% ± 18.09%                                     |
| solar_flare (361244)              | inf ± nan           | 0.6230 ± 0.2459                      | 0.7754 ± 0.1473          | 0.3677 ± 0.0618            | 58664596381642200.00% ± 14925945945379184.00%      |
| naval_propulsion_plant (361247)   | -7.3417 ± 0.0137    | 0.0000 ± 0.0000                      | 0.0001 ± 0.0000          | 0.0001 ± 0.0000            | 0.01% ± 0.00%                                       |
| white_wine (361249)               | inf ± nan           | 0.4841 ± 0.0471                      | 0.6949 ± 0.0348          | 0.5209 ± 0.0228            | 9.44% ± 0.54%                                       |
| red_wine (361250)                 | -2.6663 ± 0.1812    | 0.3901 ± 0.0467                      | 0.6234 ± 0.0383          | 0.4831 ± 0.0259            | 8.80% ± 0.74%                                       |
| Moneyball (361616)                | 4.7568 ± 0.0510     | 793.5134 ± 72.0367                   | 28.1402 ± 1.2825         | 22.1533 ± 0.8973           | 3.14% ± 0.14%                                       |
| energy_efficiency (361617)        | 0.1188 ± 0.0678     | 0.1692 ± 0.0479                      | 0.4073 ± 0.0577          | 0.2771 ± 0.0315            | 1.18% ± 0.13%                                       |
| forest_fires (361618)             | 2.1998 ± 0.3645     | 4064.6785 ± 7122.7605                 | 46.2970 ± 43.8323        | 15.0338 ± 7.0151           | 1551865282637844480.00% ± 352833180656029056.00%   |
| student_performance_por (361619)  | 0.6359 ± 0.2766     | 7.4011 ± 2.6964                      | 2.6748 ± 0.4964          | 1.9942 ± 0.2445            | 98542525092099632.00% ± 102379799818153664.00%     |
| QSAR_fish_toxicity (361621)       | 1.0693 ± 0.0887     | 0.7368 ± 0.0981                      | 0.8565 ± 0.0574          | 0.5947 ± 0.0388            | 23.73% ± 10.51%                                     |
| cars (361622)                     | 8.5861 ± 0.0723     | 4209088.5973 ± 649921.0897           | 2045.1781 ± 162.2813      | 1384.2685 ± 108.5005       | 6.39% ± 0.38%                                       |
| space_ga (361623)                 | -1.1043 ± 0.0664    | 0.0088 ± 0.0041                      | 0.0918 ± 0.0182          | 0.0647 ± 0.0042            | 12.75% ± 1.13%                                      |

- video_transcoding: error during predictions, NLL: inf and little bad regressor scores
- cpu_activity: error during predictions, but good scores
- diamonds, miami_housing, socmob, kings_county, brazilian_houses, fifa, auction_verification, solar_flare, forest_fires, cars: error during predictions, bad scores 

## OpenML-CTR23 (Pre-prcoess)

10 Folds for each dataset, below is with pre-prcoessing to the dataset (most notebly one hot encoding)

Link: https://www.kaggle.com/code/ayhamo/thesis-main?scriptVersionId=241417651

| Dataset                           | Avg Test NLL        | Avg Test MSE                         | Avg Test RMSE            | Avg Test MAE               | Avg Test MAPE                                       |
| :-------------------------------- | :------------------ | :----------------------------------- | :----------------------- | :------------------------- | :-------------------------------------------------- |
| grid_stability (361251)           | -4.1043 ± 0.0168    | 0.0000 ± 0.0000                      | 0.0055 ± 0.0003          | 0.0036 ± 0.0001            | 95.38% ± 125.49%                                    |
| video_transcoding (361252)        | inf ± nan           | 12.9445 ± 2.0629                     | 3.5859 ± 0.2932          | 0.7569 ± 0.0444            | 5.28% ± 0.06%                                       |
| wave_energy (361253)              | 9.9437 ± 0.0121     | 20177032.2934 ± 4025032.8038          | 4470.6085 ± 436.6832     | 3020.2822 ± 81.7656        | 0.08% ± 0.00%                                       |
| sarcos (361254)                   | 1.9212 ± 0.0140     | 7.7924 ± 0.9086                      | 2.7871 ± 0.1564          | 1.6249 ± 0.0299            | 993457449473995.38% ± 429951039506990.75%           |
| california_housing (361255)       | 11.5564 ± 0.0366    | 1847555008.9483 ± 191653405.8820      | 42925.4313 ± 2227.6346   | 26436.8711 ± 630.3712       | 14.24% ± 0.58%                                      |
| cpu_activity (361256)             | 1.8632 ± 0.0360     | 7.1112 ± 0.6237                      | 2.6641 ± 0.1163          | 1.7302 ± 0.0804            | 145069519335287776.00% ± 30210526094306520.00%      |
| diamonds (361257)                 | inf ± nan           | 281595.4468 ± 19083.7879             | 530.3419 ± 18.2447       | 264.0328 ± 8.4774          | 6.21% ± 0.14%                                       |
| kin8nm (361258)                   | -1.2821 ± 0.0268    | 0.0049 ± 0.0003                      | 0.0699 ± 0.0023          | 0.0545 ± 0.0021            | 10.42% ± 0.80%                                      |
| pumadyn32nh (361259)              | -2.4347 ± 0.0326    | 0.0004 ± 0.0000                      | 0.0209 ± 0.0007          | 0.0166 ± 0.0006            | 341.77% ± 214.82%                                   |
| miami_housing (361260)            | 11.8487 ± 0.0387    | 7570139372.3785 ± 618035857.1107     | 86935.5639 ± 3513.8431   | 39242.1456 ± 1296.9272      | 9.13% ± 0.38%                                       |
| cps88wages (361261)               | inf ± nan           | 146638.2187 ± 34645.6972             | 380.4742 ± 43.3313       | 220.9764 ± 3.8062          | 50.91% ± 1.36%                                      |
| socmob (361264)                   | 2.0070 ± 0.1234     | 183.6488 ± 195.4986                   | 12.0529 ± 6.1948         | 4.2007 ± 0.9983            | 43309158357709944.00% ± 10921369994031978.00%       |
| kings_county (361266)             | 12.4506 ± 0.0278    | 14132590527.7067 ± 4757652673.5136   | 117462.1375 ± 18309.4725 | 60253.3985 ± 3399.2470      | 11.28% ± 0.25%                                      |
| brazilian_houses (361267)         | inf ± nan           | 118880180.7745 ± 206370607.5488      | 7215.3342 ± 8174.2971    | 1540.2223 ± 278.9010       | 26.72% ± 0.82%                                      |
| health_insurance (361269)         | inf ± nan           | 209.2058 ± 5.5444                    | 14.4627 ± 0.1922         | 11.1888 ± 0.2007           | 2114021637106370048.00% ± 55012460508906280.00%     |
| fifa (361272)                     | 8.2709 ± 0.0526     | 108425961.4904 ± 23289032.7671       | 10351.0916 ± 1131.7528   | 3795.9989 ± 246.3346        | 82.09% ± 5.60%                                      |
| abalone (361234)                  | 0.2102 ± 0.0750     | 4.2753 ± 0.5141                      | 2.0641 ± 0.1218          | 1.4203 ± 0.0736            | 13.80% ± 0.64%                                      |
| airfoil_self_noise (361235)       | 1.0360 ± 0.0544     | 0.9484 ± 0.2265                      | 0.9666 ± 0.1188          | 0.6233 ± 0.0557            | 0.50% ± 0.04%                                       |
| auction_verification (361236)     | 6.1136 ± 0.1299     | 364900.8179 ± 92578.7975             | 599.0134 ± 77.9984       | 282.7366 ± 38.9403         | 7.44% ± 2.65%                                       |
| concrete_compressive_strength (361237) | 2.1725 ± 0.1066     | 14.9270 ± 6.7669                     | 3.7765 ± 0.8155          | 2.2283 ± 0.2765            | 7.32% ± 0.85%                                       |
| physiochemical_protein (361241)   | inf ± nan           | 12.2795 ± 0.2773                     | 3.5040 ± 0.0396          | 2.2469 ± 0.0291            | 15055142516309920.00% ± 3342818623822434.00%        |
| superconductivity (361242)        | inf ± nan           | 94.9896 ± 6.7674                     | 9.7401 ± 0.3472          | 5.4539 ± 0.1176            | 565.37% ± 611.25%                                   |
| geographical_origin_of_music (361243) | 2.2758 ± 0.2180     | 210.0773 ± 53.8257                   | 14.3860 ± 1.7668         | 9.7080 ± 1.1132            | 82.67% ± 18.09%                                     |
| solar_flare (361244)              | inf ± nan           | 0.6290 ± 0.2475                      | 0.7794 ± 0.1468          | 0.3698 ± 0.0604            | 57614126961454184.00% ± 14794070404963224.00%       |
| naval_propulsion_plant (361247)   | -7.3417 ± 0.0137    | 0.0000 ± 0.0000                      | 0.0001 ± 0.0000          | 0.0001 ± 0.0000            | 0.01% ± 0.00%                                       |
| white_wine (361249)               | inf ± nan           | 0.4841 ± 0.0471                      | 0.6949 ± 0.0348          | 0.5209 ± 0.0228            | 9.44% ± 0.54%                                       |
| red_wine (361250)                 | -2.6663 ± 0.1812    | 0.3901 ± 0.0467                      | 0.6234 ± 0.0383          | 0.4831 ± 0.0259            | 8.80% ± 0.74%                                       |
| Moneyball (361616)                | 4.4766 ± 0.0377     | 437.3483 ± 31.5674                   | 20.8995 ± 0.7481         | 16.4620 ± 0.7373           | 2.32% ± 0.10%                                       |
| energy_efficiency (361617)        | 0.1188 ± 0.0678     | 0.1692 ± 0.0479                      | 0.4073 ± 0.0577          | 0.2771 ± 0.0315            | 1.18% ± 0.13%                                       |
| forest_fires (361618)             | 2.1972 ± 0.3690     | 4063.1253 ± 7119.5515                | 46.2842 ± 43.8281        | 14.9761 ± 6.9952           | 1524105306553441792.00% ± 353001698688530880.00%    |
| student_performance_por (361619)  | 0.6457 ± 0.2869     | 7.4567 ± 2.7828                      | 2.6828 ± 0.5091          | 1.9960 ± 0.2458            | 100963720966075952.00% ± 105081337274491376.00%     |
| QSAR_fish_toxicity (361621)       | 1.0693 ± 0.0887     | 0.7368 ± 0.0981                      | 0.8565 ± 0.0574          | 0.5947 ± 0.0388            | 23.73% ± 10.51%                                     |
| cars (361622)                     | 8.5861 ± 0.0723     | 4209088.5973 ± 649921.0897           | 2045.1781 ± 162.2813     | 1384.2685 ± 108.5005       | 6.39% ± 0.38%                                       |
| space_ga (361623)                 | -1.1043 ± 0.0664    | 0.0088 ± 0.0041                      | 0.0918 ± 0.0182          | 0.0647 ± 0.0042            | 12.75% ± 1.13%                                      |


# Summary on OpenML-CTR23 pre-prcoess vs not:
| Dataset                           | Effect of Pre-processing                                                                         |
| :-------------------------------- | :----------------------------------------------------------------------------------------------- |
| grid_stability (361251)           | No change across all metrics.                                                                    |
| video_transcoding (361252)        | Worsened MSE, RMSE, and slightly worsened MAE, but improved MAPE. NLL remained inf.                |
| wave_energy (361253)              | No change across all metrics.                                                                    |
| sarcos (361254)                   | No change across all metrics.                                                                    |
| california_housing (361255)       | No change across all metrics.                                                                    |
| cpu_activity (361256)             | No change across all metrics.                                                                    |
| diamonds (361257)                 | Improved MSE and RMSE, but slightly worsened MAE and worsened MAPE. NLL remained inf.              |
| kin8nm (361258)                   | No change across all metrics.                                                                    |
| pumadyn32nh (361259)              | No change across all metrics.                                                                    |
| miami_housing (361260)            | No change across all metrics.                                                                    |
| cps88wages (361261)               | Slightly worsened MSE, RMSE, and MAE, but slightly improved MAPE. NLL remained inf.              |
| socmob (361264)                   | Consistently improved all metrics.                                                               |
| kings_county (361266)             | Worsened NLL, MAE, and MAPE, but improved MSE and RMSE.                                          |
| brazilian_houses (361267)         | Consistently worsened all metrics where comparison was possible (MSE, RMSE, MAE, MAPE). NLL remained inf. |
| health_insurance (361269)         | Improved MSE and RMSE, but worsened MAE and MAPE. NLL remained inf.                              |
| fifa (361272)                     | Consistently improved all metrics.                                                               |
| abalone (361234)                  | Worsened NLL, MSE, and RMSE, but improved MAE and MAPE.                                          |
| airfoil_self_noise (361235)       | No change across all metrics.                                                                    |
| auction_verification (361236)     | Significantly improved all metrics.                                                              |
| concrete_compressive_strength (361237) | No change across all metrics.                                                                    |
| physiochemical_protein (361241)   | No change across all metrics (NLL remained inf, other comparable metrics were identical).          |
| superconductivity (361242)        | No change across all metrics (NLL remained inf, other comparable metrics were identical).          |
| geographical_origin_of_music (361243) | No change across all metrics.                                                                    |
| solar_flare (361244)              | Worsened MSE, RMSE, and MAE, but improved MAPE. NLL remained inf.                                |
| naval_propulsion_plant (361247)   | No change across all metrics.                                                                    |
| white_wine (361249)               | No change across all metrics (NLL remained inf, other comparable metrics were identical).          |
| red_wine (361250)                 | No change across all metrics.                                                                    |
| Moneyball (361616)                | Consistently improved all metrics.                                                               |
| energy_efficiency (361617)        | No change across all metrics.                                                                    |
| forest_fires (361618)             | Consistently (slightly) improved all metrics.                                                    |
| student_performance_por (361619)  | Consistently worsened all metrics.                                                               |
| QSAR_fish_toxicity (361621)       | No change across all metrics.                                                                    |
| cars (361622)                     | No change across all metrics.                                                                    |
| space_ga (361623)                 | No change across all metrics.                                                                    |