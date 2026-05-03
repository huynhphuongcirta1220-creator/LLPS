# Baseline residual diagnostics summary

## Protocol metrics

| protocol                    |   n_rows |   n_groups |   auc_mean |   auc_std |   global_auc |       f1 |   precision |   recall |   tn |   fp |   fn |   tp |
|:----------------------------|---------:|-----------:|-----------:|----------:|-------------:|---------:|------------:|---------:|-----:|-----:|-----:|-----:|
| DS-known                    |      493 |        150 |   0.831983 | 0.0880317 |     0.828956 | 0.833083 |    0.814706 | 0.852308 |  105 |   63 |   48 |  277 |
| DS-unknown                  |      493 |         69 |   0.839719 | 0.0483036 |     0.794277 | 0.82389  |    0.820122 | 0.827692 |  109 |   59 |   56 |  269 |
| LOCO                        |      493 |         27 |   0.894798 | 0.126763  |     0.74544  | 0.809091 |    0.797015 | 0.821538 |  100 |   68 |   58 |  267 |
| strict_amine_object_holdout |      352 |         21 |   0.873622 | 0.127834  |     0.71643  | 0.795745 |    0.799145 | 0.792373 |   69 |   47 |   49 |  187 |


## DS-unknown

### Top underpredicted canonical amine objects

- 2MPRZ | n=6 | residual_mean=0.586 | abs_mean=0.586 | fn_pos=0.833 | fp_neg=nan
- AEP | n=28 | residual_mean=0.380 | abs_mean=0.467 | fn_pos=0.520 | fp_neg=0.333
- DEEA|TETA | n=6 | residual_mean=0.263 | abs_mean=0.263 | fn_pos=0.000 | fp_neg=nan
- DETA|PMDETA | n=17 | residual_mean=0.222 | abs_mean=0.222 | fn_pos=0.118 | fp_neg=nan
- MEA | n=121 | residual_mean=0.199 | abs_mean=0.337 | fn_pos=0.242 | fp_neg=0.200
- EAE | n=23 | residual_mean=0.168 | abs_mean=0.221 | fn_pos=0.105 | fp_neg=0.000

### Top overpredicted canonical amine objects

- DEEA|HMDA | n=13 | residual_mean=-0.471 | abs_mean=0.471 | fn_pos=nan | fp_neg=0.308
- BZMEA | n=5 | residual_mean=-0.346 | abs_mean=0.346 | fn_pos=nan | fp_neg=0.000
- MAE | n=69 | residual_mean=-0.334 | abs_mean=0.449 | fn_pos=0.036 | fp_neg=0.707
- BAE | n=6 | residual_mean=-0.270 | abs_mean=0.270 | fn_pos=nan | fp_neg=0.167
- DMAE|TETA | n=27 | residual_mean=-0.230 | abs_mean=0.230 | fn_pos=nan | fp_neg=0.000
- IPMEA | n=5 | residual_mean=-0.147 | abs_mean=0.147 | fn_pos=nan | fp_neg=0.000

### Solvent-side bias hotspots

- TGBE | n=6 | residual_mean=0.586 | abs_mean=0.586 | fn_pos=0.833 | fp_neg=nan
- TEGDME | n=5 | residual_mean=0.538 | abs_mean=0.917 | fn_pos=1.000 | fp_neg=1.000
- sulfolane | n=40 | residual_mean=0.416 | abs_mean=0.420 | fn_pos=0.500 | fp_neg=0.000
- isobutanol | n=5 | residual_mean=0.288 | abs_mean=0.288 | fn_pos=0.200 | fp_neg=nan
- Tertiarybutanol | n=6 | residual_mean=0.236 | abs_mean=0.261 | fn_pos=0.200 | fp_neg=0.000
- DEGDEE | n=23 | residual_mean=0.168 | abs_mean=0.235 | fn_pos=0.100 | fp_neg=0.000

- ethanol | n=14 | residual_mean=-0.410 | abs_mean=0.521 | fn_pos=0.500 | fp_neg=0.583
- Methanol | n=6 | residual_mean=-0.333 | abs_mean=0.333 | fn_pos=nan | fp_neg=0.167
- EGBE | n=12 | residual_mean=-0.223 | abs_mean=0.390 | fn_pos=0.500 | fp_neg=0.300
- DEGEE | n=15 | residual_mean=-0.176 | abs_mean=0.304 | fn_pos=0.000 | fp_neg=0.200
- (none) | n=132 | residual_mean=-0.104 | abs_mean=0.300 | fn_pos=0.074 | fp_neg=0.297
- DGME | n=19 | residual_mean=-0.022 | abs_mean=0.225 | fn_pos=0.100 | fp_neg=0.111

### Physical bins with largest bias

- under / T_bin: <=303 K (res=0.168, n=132); 313–333 K (res=-0.017, n=45); 303–313 K (res=-0.032, n=268)
- over  / T_bin: >333 K (res=-0.043, n=48); 303–313 K (res=-0.032, n=268); 313–333 K (res=-0.017, n=45)
- under / H2O_wt_bin: 0 (res=0.128, n=142); 30–60 (res=0.054, n=124); 10–30 (res=-0.011, n=107)
- over  / H2O_wt_bin: >60 (res=-0.157, n=78); 0–10 (res=-0.014, n=42); 10–30 (res=-0.011, n=107)
- under / total_amine_wt_bin: 30–45 (res=0.113, n=63); >45 (res=0.086, n=107); <=15 (res=0.010, n=31)
- over  / total_amine_wt_bin: 15–30 (res=-0.020, n=292); <=15 (res=0.010, n=31); >45 (res=0.086, n=107)
- under / Delta_S_np_qbin: Delta_S_np1: 0.00033 ~ 0.0647 (res=0.105, n=102); Delta_S_np4: 0.288 ~ 0.512 (res=0.059, n=93); Delta_S_np3: 0.165 ~ 0.288 (res=0.030, n=92)
- over  / Delta_S_np_qbin: Delta_S_np5: 0.512 ~ 0.677 (res=-0.074, n=121); Delta_S_np2: 0.0647 ~ 0.165 (res=0.010, n=85); Delta_S_np3: 0.165 ~ 0.288 (res=0.030, n=92)
- under / Delta_S_acc_qbin: Delta_S_acc3: 0.0867 ~ 0.137 (res=0.061, n=88); Delta_S_acc2: 0.0373 ~ 0.0867 (res=0.050, n=98); Delta_S_acc4: 0.137 ~ 0.223 (res=0.048, n=94)
- over  / Delta_S_acc_qbin: Delta_S_acc5: 0.223 ~ 0.324 (res=-0.066, n=120); Delta_S_acc1: -0.000893 ~ 0.0373 (res=0.044, n=93); Delta_S_acc4: 0.137 ~ 0.223 (res=0.048, n=94)
- under / Sal_Out_qbin: Sal_Out1: -0.001 ~ 0.22 (res=0.091, n=197); Sal_Out4: 0.339 ~ 0.466 (res=0.051, n=91); Sal_Out3: 0.282 ~ 0.339 (res=0.018, n=94)
- over  / Sal_Out_qbin: Sal_Out2: 0.22 ~ 0.282 (res=-0.120, n=111); Sal_Out3: 0.282 ~ 0.339 (res=0.018, n=94); Sal_Out4: 0.339 ~ 0.466 (res=0.051, n=91)

## strict_amine_object_holdout

### Top underpredicted canonical amine objects

- 2MPRZ | n=6 | residual_mean=0.427 | abs_mean=0.427 | fn_pos=0.167 | fp_neg=nan
- AEP | n=28 | residual_mean=0.426 | abs_mean=0.508 | fn_pos=0.480 | fp_neg=0.333
- DAP | n=5 | residual_mean=0.415 | abs_mean=0.415 | fn_pos=0.400 | fp_neg=nan
- MEA | n=121 | residual_mean=0.234 | abs_mean=0.351 | fn_pos=0.286 | fp_neg=0.133
- EAE | n=23 | residual_mean=0.197 | abs_mean=0.242 | fn_pos=0.105 | fp_neg=0.000
- DETA | n=28 | residual_mean=0.067 | abs_mean=0.266 | fn_pos=0.050 | fp_neg=0.125

### Top overpredicted canonical amine objects

- BZMEA | n=5 | residual_mean=-0.669 | abs_mean=0.669 | fn_pos=nan | fp_neg=1.000
- MAE | n=69 | residual_mean=-0.389 | abs_mean=0.459 | fn_pos=0.000 | fp_neg=0.756
- BAE | n=6 | residual_mean=-0.275 | abs_mean=0.275 | fn_pos=nan | fp_neg=0.167
- IPMEA | n=5 | residual_mean=-0.151 | abs_mean=0.151 | fn_pos=nan | fp_neg=0.000
- TEPA | n=16 | residual_mean=-0.005 | abs_mean=0.165 | fn_pos=0.000 | fp_neg=0.500
- DEA | n=12 | residual_mean=0.023 | abs_mean=0.023 | fn_pos=0.000 | fp_neg=nan

### Solvent-side bias hotspots

- TEGDME | n=5 | residual_mean=0.555 | abs_mean=0.952 | fn_pos=1.000 | fp_neg=1.000
- sulfolane | n=35 | residual_mean=0.534 | abs_mean=0.538 | fn_pos=0.576 | fp_neg=0.000
- TGBE | n=6 | residual_mean=0.427 | abs_mean=0.427 | fn_pos=0.167 | fp_neg=nan
- isobutanol | n=5 | residual_mean=0.311 | abs_mean=0.311 | fn_pos=0.200 | fp_neg=nan
- Tertiarybutanol | n=6 | residual_mean=0.252 | abs_mean=0.274 | fn_pos=0.200 | fp_neg=0.000
- DEGDEE | n=23 | residual_mean=0.176 | abs_mean=0.266 | fn_pos=0.100 | fp_neg=0.333

- ethanol | n=14 | residual_mean=-0.482 | abs_mean=0.575 | fn_pos=0.500 | fp_neg=0.667
- (none) | n=12 | residual_mean=-0.358 | abs_mean=0.358 | fn_pos=nan | fp_neg=0.167
- Methanol | n=6 | residual_mean=-0.296 | abs_mean=0.296 | fn_pos=nan | fp_neg=0.167
- EGBE | n=12 | residual_mean=-0.218 | abs_mean=0.379 | fn_pos=0.000 | fp_neg=0.300
- DEGEE | n=15 | residual_mean=-0.178 | abs_mean=0.304 | fn_pos=0.000 | fp_neg=0.300
- DGME | n=19 | residual_mean=-0.083 | abs_mean=0.199 | fn_pos=0.000 | fp_neg=0.222

### Physical bins with largest bias

- under / T_bin: <=303 K (res=0.242, n=104); 313–333 K (res=0.211, n=14); >333 K (res=0.063, n=31)
- over  / T_bin: 303–313 K (res=-0.049, n=203); >333 K (res=0.063, n=31); 313–333 K (res=0.211, n=14)
- under / H2O_wt_bin: 0 (res=0.163, n=128); 30–60 (res=0.044, n=98); 10–30 (res=0.001, n=65)
- over  / H2O_wt_bin: >60 (res=-0.216, n=19); 0–10 (res=-0.025, n=42); 10–30 (res=0.001, n=65)
- under / total_amine_wt_bin: >45 (res=0.384, n=31); <=15 (res=0.250, n=17); 30–45 (res=0.067, n=38)
- over  / total_amine_wt_bin: 15–30 (res=0.005, n=266); 30–45 (res=0.067, n=38); <=15 (res=0.250, n=17)
- under / Delta_S_np_qbin: Delta_S_np1: 0.00033 ~ 0.0647 (res=0.117, n=99); Delta_S_np4: 0.288 ~ 0.512 (res=0.053, n=85); Delta_S_np2: 0.0647 ~ 0.165 (res=0.035, n=74)
- over  / Delta_S_np_qbin: Delta_S_np3: 0.165 ~ 0.288 (res=0.025, n=92); Delta_S_np2: 0.0647 ~ 0.165 (res=0.035, n=74); Delta_S_np4: 0.288 ~ 0.512 (res=0.053, n=85)
- under / Delta_S_acc_qbin: Delta_S_acc1: -0.000893 ~ 0.0373 (res=0.066, n=93); Delta_S_acc3: 0.0867 ~ 0.137 (res=0.064, n=82); Delta_S_acc2: 0.0373 ~ 0.0867 (res=0.061, n=90)
- over  / Delta_S_acc_qbin: Delta_S_acc4: 0.137 ~ 0.223 (res=0.047, n=83); Delta_S_acc2: 0.0373 ~ 0.0867 (res=0.061, n=90); Delta_S_acc3: 0.0867 ~ 0.137 (res=0.064, n=82)
- under / Sal_Out_qbin: Sal_Out1: -0.001 ~ 0.22 (res=0.146, n=144); Sal_Out4: 0.339 ~ 0.466 (res=0.053, n=91); Sal_Out3: 0.282 ~ 0.339 (res=0.030, n=83)
- over  / Sal_Out_qbin: Sal_Out2: 0.22 ~ 0.282 (res=-0.244, n=34); Sal_Out3: 0.282 ~ 0.339 (res=0.030, n=83); Sal_Out4: 0.339 ~ 0.466 (res=0.053, n=91)

## Cross-protocol persistent bias (DS-unknown ∩ strict)

### Consistent strong underprediction candidates

- AMP | n=4 | strict_mean=0.942 | dsu_mean=0.910 | share_strong_under=1.000
- 2MPRZ | n=6 | strict_mean=0.427 | dsu_mean=0.586 | share_strong_under=0.833
- AEP | n=28 | strict_mean=0.426 | dsu_mean=0.380 | share_strong_under=0.500
- MEA | n=121 | strict_mean=0.234 | dsu_mean=0.199 | share_strong_under=0.438
- 1-amino-2-propanol | n=8 | strict_mean=0.052 | dsu_mean=0.075 | share_strong_under=0.250
- DETA | n=28 | strict_mean=0.067 | dsu_mean=-0.015 | share_strong_under=0.143
- EAE | n=23 | strict_mean=0.197 | dsu_mean=0.168 | share_strong_under=0.130
- TEPA | n=16 | strict_mean=-0.005 | dsu_mean=0.081 | share_strong_under=0.062

### Consistent strong overprediction candidates

- AMB | n=1 | strict_mean=-0.885 | dsu_mean=-0.913 | share_strong_over=1.000
- DEEA | n=1 | strict_mean=-0.725 | dsu_mean=-0.274 | share_strong_over=1.000
- BZMEA | n=5 | strict_mean=-0.669 | dsu_mean=-0.346 | share_strong_over=1.000
- MPA | n=3 | strict_mean=-0.440 | dsu_mean=-0.445 | share_strong_over=1.000
- MAE | n=69 | strict_mean=-0.389 | dsu_mean=-0.334 | share_strong_over=0.522
- DETA | n=28 | strict_mean=0.067 | dsu_mean=-0.015 | share_strong_over=0.250
- BAE | n=6 | strict_mean=-0.275 | dsu_mean=-0.270 | share_strong_over=0.167
- 1-amino-2-propanol | n=8 | strict_mean=0.052 | dsu_mean=0.075 | share_strong_over=0.125
