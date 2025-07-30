import pandas as pd
import numpy as np
import os
import Utils
from scipy import stats
from scipy.interpolate import interp1d

def read_raw_csv(filepath):
    """
    Read the data results of a single CSV file and extract the required data (DOY days and radar signals).
    :param filepath:
    :return: The dataframe of the required radar signal, column named DOY days.
    """
    raw_csv_data = pd.read_csv(filepath)
    radar_temporal = raw_csv_data.loc[:, "VH"]
    radar_temporal.index = raw_csv_data.loc[:, "DOY"]
    radar_temporal.name = os.path.basename(filepath)
    
    return radar_temporal


def get_all_radar_csv_data(folder_path, crop):
    """
    :param folder_path: Read folders
    :param crop: Crop category
    :return: Returns a CSV file consisting of all radar time series data
    """
    files_with_corn = []
    file_names = []
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if crop in filename:
                files_with_corn.append(os.path.join(folder_path, filename))

    radar_temporal_list = []
    for csv_path in files_with_corn:
        radar_temporal_list.append(read_raw_csv(csv_path))
        print(csv_path)
        file_names.append(os.path.basename(csv_path))
        
    return pd.DataFrame(radar_temporal_list),file_names


def process_doy_columns_processed(rawDf):
    """
    Read the CSV file, filter out the columns containing 'Doy', remove the 'Doy_' prefix and convert to an integer index,
    Finally, the DataFrame is returned by column index sorting.

    Parameter:
    filepath (str): The path to the CSV file.

    Return:
    DataFrame: The processed DataFrame contains only the columns with 
    'Doy' in the column name, and the column name is an integer index.
    """

    rawDf.index = rawDf['index']
    rawDf = rawDf[[col for col in rawDf.columns if 'Doy' in col]]
    rawDf = rawDf.rename(columns=lambda x: int(x.replace('Doy_', '')))
    rawDf = rawDf.sort_index(axis=1)

    return rawDf

def shift_and_fill(df):
    """
    Perform an offset in the column direction for a given DataFrame and 
    process the NaN value using forward or backward padding depending on the direction of the displacement.

    Parameter:
    eachWindowData_df_filtered (pd. DataFrame): The DataFrame that needs to be processed.

    Return:
    list of pd. DataFrame: A list of DataFrames for each offset version.
    """
    eachWindow_df_filtered_shift_list = []
    for i in range(-30, 31, 1):
        eachWindowData_df_filtered_shifted = df.shift(i, axis=1)
        if i < 0:
            eachWindowData_df_filtered_shifted = eachWindowData_df_filtered_shifted.ffill(axis=1)
        elif i > 0:
            eachWindowData_df_filtered_shifted = eachWindowData_df_filtered_shifted.bfill(axis=1)

        eachWindow_df_filtered_shift_list.append(eachWindowData_df_filtered_shifted)

    return eachWindow_df_filtered_shift_list


def find_min_msd_row(series, df):
    """
    Calculate the mean square deviation (MSD) for each row in a Series versus DataFrame and 
    return the row with the smallest MSD (in Series form).

    :param series: The series of the input
    :param df: The DataFrame of the input
    :return: MSD smallest line (in Series form)
    """
    def msd(row, series):
        return np.mean((row - series) ** 2)
    msd_values = df.apply(lambda row: msd(row, series), axis=1)
    min_msd_index = msd_values.idxmin()
    min_msd_row = df.loc[min_msd_index]

    return min_msd_row


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


def find_max_r_row(series, df):
    """
    Calculate the mean square deviation (MSD) for each row in a Series versus DataFrame and 
    return the row with the smallest MSD (in Series form).

    :param series: The series of the input
    :param df: The DataFrame of the input
    :return: R largest row (in Series form)
    """
    r_values = df.apply(lambda row: calculate_pearson_coefficient(row, series), axis=1)
    if r_values.empty or r_values.isna().all():
            return None  
    max_r_index = r_values.idxmax()
    max_r_row = df.loc[max_r_index]

    return max_r_row


def interpolate_dataframe(df):
    x_new = np.arange(1, 362)
    new_data = []
    for idx, row in df.iterrows():

        f = interp1d(df.columns, row.values, kind='linear', fill_value='extrapolate')
        y_new = f(x_new)
        new_data.append(y_new)
    new_df = pd.DataFrame(new_data, columns=x_new, index=df.index)
    return new_df

### This section does the first stretch of the data
if __name__ == "__main__":
    vege_type = "Soybean"
    base_path = r'your file path~\Phenocam site'
    folder_path = os.path.join(base_path,'Raw Phenocam site timing data')
    cropDf,order_filenames = get_all_radar_csv_data(folder_path, vege_type)

    raw_df_smoothed = cropDf.apply(lambda row: Utils.moving_average_filter(row), axis=1)
    raw_df_smoothed = interpolate_dataframe(raw_df_smoothed)
    medianFolder = os.path.join(base_path,'Multi-year average Phenocam sampling results')
    output_path = os.path.join(os.path.dirname(medianFolder), "Global stretching")
    os.makedirs(output_path, exist_ok=True)
    out_df = pd.DataFrame(index=np.arange(start=0, stop=len(cropDf.index)), columns=raw_df_smoothed.columns)
    for i, (index, row) in enumerate(raw_df_smoothed.iterrows()):
        print(index)
        raw_name = order_filenames[i]
        print(raw_name)
        if index != raw_name:
            raise ValueError(f"index and raw_name do not match: {index} != {raw_name}")
        median_name = raw_name.replace('rawOutput', 'MedianData_', 1)
        median_df = pd.read_csv(
            os.path.join(medianFolder, median_name),
            index_col=0)
        median_df.columns = np.arange(start=4, step=6, stop=366)

        median_df = median_df.apply(lambda r: Utils.moving_average_filter(r), axis=1)
        median_df = interpolate_dataframe(median_df)
        eachWindow_df_filtered_shift_list = shift_and_fill(median_df)
        eachWindow_df_filtered_stretched = []
        for shift_df in eachWindow_df_filtered_shift_list:
            for sigma in [0.9, 0.95, 1.0, 1.05, 1.1]:
                if sigma == 1:
                    eachWindow_df_filtered_stretched.append(shift_df)
                else:
                    eachWindow_df_filtered_stretched.append(
                        Utils.time_stretch(shift_df, sigma))
        merged_df = pd.concat(eachWindow_df_filtered_stretched, ignore_index=True)
        best_res = find_max_r_row(row, merged_df)
        gmfr_res, _ = Utils.calculate_gmfr(row, best_res)
        out_df.loc[i] = gmfr_res
        out_df.at[i, 'raw_name'] = raw_name
        print(raw_name, "数据已经处理完成！")

    output_file = f'{vege_type}Phenocam post-processing data (global stretch translation).csv'
    full_output_path = os.path.join(output_path, output_file)
    os.makedirs(output_path, exist_ok=True)

    out_df.to_csv(full_output_path, index=True, encoding="utf-8")
    print(f"The data is saved to：{full_output_path}")
    print("The processed file has been output！")
