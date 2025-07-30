import os
import pandas as pd
import numpy as np
import Utils
from scipy.interpolate import interp1d
from scipy import stats

def read_raw_csv(filepath):
    """
    Read the data results of a single CSV file and extract the required data (DOY days and radar signals).
    :param filepath: CSV file path
    :return: A row of a one-dimensional array or DataFrame of the required radar signal, with the column name DOY days.
    """
    raw_csv_data = pd.read_csv(filepath)
    vh_data = raw_csv_data['VH'].values
    doy_columns = raw_csv_data['DOY'].values
    radar_temporal = pd.DataFrame([vh_data], columns=doy_columns)
    return radar_temporal
def sort_key(filename):
    file_name = filename.split('\\')[-1]
    base_name = file_name.split('.')[0]
    parts = base_name.split('_')

    year = int(parts[-2])  
    index = int(parts[-1])  

    return (year, index)  
def get_all_radar_csv_data(folder_path, crop):
     """
    :param folder_path: Read folders
    :param crop: Crop category
    :return: Returns a CSV file consisting of all radar time series data
     """
     files_with_corn = []
     for dirpath, dirnames, filenames in os.walk(folder_path):
         for filename in filenames:
             if crop in filename:
                 files_with_corn.append(os.path.join(folder_path, filename))
     files_with_corn = sorted(files_with_corn, key=sort_key)
     radar_temporal_list = []
     for csv_path in files_with_corn:
         radar_temporal_list.append(read_raw_csv(csv_path))

     return pd.concat(radar_temporal_list, axis=0)

def time_stretch(df, scale_factor):
    """
    Scales the values of each row of the DataFrame and calculates the median value by a specific range.

    Parameter:
    df (pandas. DataFrame): A DataFrame containing doy columns (i.e., the column name is doy values) and corresponding values
    scale_factor (float): Scale factor
    end(int): The end of the grouping scope and bin tag

    Return:
    pandas. DataFrame: The new DataFrame after calculating the median
    """

    scaled_doy = np.round(df.columns.astype(float) * scale_factor, 2)
    values = df.values

    f_linear = interp1d(scaled_doy, values, kind='linear', bounds_error=False, fill_value=[np.nan])
    y_new = f_linear(df.columns)

    result_series = pd.Series(data=y_new.flatten(), index=df.columns)
    result_series = result_series.fillna(method='ffill')
    return result_series


def calculate_pearson_coefficient(x, y):
    """
    Use scipy to calculate the Pearson correlation coefficient (R-coefficient) for two variables x and y.

    Parameter:
    x (list or array): A list of cases for the first variable.
    y (list or array): A list of cases for the second variable.

    Return:
    float: Pearson correlation coefficient.
    """
    if len(x) != len(y):
        raise ValueError("输入列表长度必须一致。")
    r, p_value = stats.pearsonr(x, y)
    return r


def calculate_msd(series_a, series_b):
    """Calculate mean square difference between two series"""

    values_a = series_a.values
    values_b = series_b.values
    return ((values_a - values_b) ** 2).mean()

### Perform a second stretch match on the data
if __name__ == "__main__":
    crop_type = "Soybean"

    base_path = r'your file path~\Phenocam site'
    rawFolder = os.path.join(base_path,'Raw Phenocam site timing data')

    output_path = os.path.join(base_path,'Average results over many years after two stretches')
    os.makedirs(output_path, exist_ok=True)

    radar_data_multiyears = get_all_radar_csv_data(rawFolder, crop_type).reset_index(drop=True)
    raw_dfs = Utils.interpolate_doy_data(radar_data_multiyears)
    print(raw_dfs.shape)

    raw_dfs = raw_dfs.apply(lambda row: Utils.apply_lowess_to_row(row, 0.3), axis=1)
    raw_dfs = raw_dfs.apply(Utils.smooth_crop_offseason_signal, axis=1)

    processedDf = pd.read_csv(
        os.path.join(base_path,"Global stretching", rf"{crop_type}Phenocam post-processing data (global stretch translation).csv"),
        index_col=0)

    raw_name_col = processedDf.iloc[:, -1]  
    order_median = pd.DataFrame({
        'index': processedDf.index,  
        'raw_name': raw_name_col.values  
    })

    processedDf = processedDf.iloc[:, :361]
    processedDf.columns = np.arange(1, 362)
    processedDf.columns = np.arange(start=1, stop=362)

    processedDf = processedDf.apply(lambda row: Utils.apply_lowess_to_row(row, 0.3), axis=1)
    processedDf = processedDf.apply(Utils.smooth_crop_offseason_signal, axis=1)

    out_df = pd.DataFrame(index=processedDf.index, columns=processedDf.columns)
    processed_max_columns = processedDf.idxmax(axis=1)

    before_scale_list = []
    after_scale_list = []
    scale_paras = np.arange(start=0.8, stop=1.21, step=0.05)
    for index, ele in processed_max_columns.items():
        if ele <= 60 or ele >= 335:
            out_df.loc[index] = processedDf.loc[index]
            continue
        processed_row = processedDf.iloc[index, :]
        raw_row = raw_dfs.iloc[index, :]
        raw_row_before = raw_row.loc[1:ele].iloc[::-1]
        raw_row_behind = raw_row.loc[ele:361]
        processed_row_before = processed_row.loc[1:ele]
        processed_row_behind = processed_row.loc[ele:361]
        processed_row_before_df = pd.DataFrame(np.flip(processed_row_before.values),
                                               index=(processed_row_before.index - 1)).T
        processed_row_behind_df = pd.DataFrame(processed_row_behind.values,
                                               index=processed_row_behind.index - processed_row_behind.index[
                                                   0]).T
        stretched_list_before = []
        stretched_list_after = []
        for sigma in np.arange(start=0.8, step=0.05, stop=1.21):
            if sigma == 1:
                stretched_list_before.append(processed_row_before_df.iloc[0])
                stretched_list_after.append(processed_row_behind_df.iloc[0])
            else:
                stretched_list_before.append(
                    time_stretch(processed_row_before_df, sigma))
                stretched_list_after.append(
                    time_stretch(processed_row_behind_df, sigma))
        best_match_series_before = None
        max_r = -2
        scale_num = np.nan
        for i, series in enumerate(stretched_list_before):
            current_r = calculate_pearson_coefficient(raw_row_before, series)
            if current_r > max_r or max_r == -2:
                max_r = current_r
                best_match_series_before = series
                scale_num = scale_paras[i]
        before_scale_list.append(scale_num)
        best_match_series_after = None
        max_r = -2
        scale_num = np.nan
        for i, series in enumerate(stretched_list_after):
            current_r = calculate_pearson_coefficient(raw_row_behind, series)
            if current_r > max_r or max_r == -2:
                max_r = current_r
                best_match_series_after = series
                scale_num = scale_paras[i]
        after_scale_list.append(scale_num)

        merged_temporal = np.concatenate([best_match_series_before.iloc[::-1].values,
                                          best_match_series_after.iloc[1:].values])
        out_df.loc[index] = merged_temporal
    out_df.to_csv(
        os.path.join(output_path, f"{crop_type}Phenocam post-processing data (two matches).csv"))
    print(f'The result is saved to:{os.path.join(output_path, f"{crop_type}Phenocam post-processing data (two matches).csv")}')
