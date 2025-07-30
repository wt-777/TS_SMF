import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import Utils
### This part performs the TS-SMF algorithm to detect phenology
if __name__ == '__main__':

    crop_type = "Soybean"
    base_path = r'your file path~\Phenocam site'
    output_path = os.path.join(base_path,'Phenological detection results')
    os.makedirs(output_path, exist_ok=True)
    radar_temporals = pd.read_csv(
        os.path.join(base_path,"Average results over many years after two stretches", 
                     rf'{crop_type}Phenocam post-processing data (two matches).csv'),
        index_col=0)
    radar_temporals.columns = np.arange(1, 362)
    reference_data = pd.read_excel(
        r'your file path~\best_refenrence_curve.xlsx',
        index_col=0)
    reference_curve = Utils.calculate_mean_values(reference_data,[0])
    reference_curve = pd.DataFrame(reference_curve).T
    total_labels, reference_labels = Utils.obtain_labels(crop_type)

    reference_labels = reference_labels.round()
    ## Remove the reference site
    total_labels = total_labels.drop([20])
    radar_temporals = radar_temporals.drop([20])

    print("The preparatory processing has been completed and the SMF algorithm probing begins")


    """Direct output calculation results"""
    smf_res = Utils.smf_detect(radar_temporals, reference_curve, reference_labels)
    smfs_res = Utils.smfs_detect(radar_temporals, reference_curve, reference_labels).replace(0, np.nan)
    smfs_res = smfs_res.fillna(smf_res)
    print("The SMF-S algorithm probing is complete")
    SMFS_cal_res = Utils.calculate_rmse_mae(smfs_res, total_labels)

    print(f"SMF-S result：{SMFS_cal_res}")
    SMFS_cal_res = SMFS_cal_res.loc[["MAE"], :]
    SMFS_cal_res['MAE_mean'] = SMFS_cal_res.mean(axis=1)
    print(SMFS_cal_res)

    """Output the phenological detection results"""
    with pd.ExcelWriter(os.path.join(output_path, f"PhenocamDetect{crop_type}Res(2StepSMF).xlsx"), engine='openpyxl',
                         mode='w') as writer:
         smfs_res.to_excel(writer, sheet_name=f"{crop_type}SMF-S_Res", index=False)
    print("Phenological detection is completed")
