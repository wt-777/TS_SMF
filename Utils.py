import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
from fastdtw import fastdtw
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from datetime import datetime, timedelta

def calculate_mean_values(df, rows_to_average):
    """
    Calculate the mean of the index for the specified rows

    Parameter:
    df (pd. DataFrame): The input Pandas data frame.
    rows_to_average (list): A list of row indexes for which the mean needs to be calculated.
    """
    mean_values = df.loc[rows_to_average].mean()
    return mean_values


def obtain_labels(crop):
    
    if crop == "Corn":
        file_path = r"your file path~\maize_multiyear.xls"
        list = [3, 4, 5, 8]
    else:
        file_path = r"your file path~\soy_multiyear.xls"
        list = [3, 4, 7]

    dataframes = []
    xls = pd.ExcelFile(file_path)
   #Get the real phenological period for each site
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name, usecols=list)
        dataframes.append(df)
    combined_df = pd.concat(dataframes, ignore_index=True).replace(0, np.nan)

    ## The phenological period of the reference curve is obtained
    reference_data = pd.read_excel(r'your file path~\best_model_Soybean.xlsx',)
    reference_phenology = calculate_mean_values(reference_data, [0])
    return combined_df, reference_phenology


def interpolate_doy_data(df, new_doy_range=(1, 361)):
    """
    Linear interpolation of each row of data in the DataFrame extends its DOY range to the specified new_doy_range.
    """

    start, end = new_doy_range
    interpolated_df = pd.DataFrame(index=df.index, columns=range(start, end + 1))
    for index, row in df.iterrows():
        original_doy = np.array(list(map(int, row.index)))
        original_values = row.values
        #Linear interpolation
        f = interp1d(original_doy, original_values, kind='linear', fill_value="extrapolate")
        interpolated_values = f(range(start, end + 1))
        interpolated_df.loc[index] = interpolated_values

    return interpolated_df


# Phenology detection algorithm
"""SMFS Algorithm"""
def smfs_detect(temporal_df, reference_curve_, reference_phenology):
    smfs_res = pd.DataFrame(index=temporal_df.index, columns=reference_phenology.index)
    for index, row in temporal_df.iterrows():
        smfs_res.loc[index] = smfs_1curve(reference_curve_.iloc[0].values, row.values, reference_phenology.values)

    return smfs_res


def smfs_1curve(shape_model, sample_curve, shape_model_point):
    shape_model_spl_ex = shape_model  
    max_num = shape_model_spl_ex.shape[0]  
    sample_curve_spl_ex = sample_curve  
    output = np.zeros(len(shape_model_point))  

    for point in range(len(shape_model_point)):  
        p0 = shape_model_point[point]  
        # method
        w = 60  
        for q in range(50):
            if q == 0:
                xscale = 1

            # tshift
            R = np.zeros(61)
            for t in range(-30, 31):
                num_begin = int(p0 - t - w)
                num_end = int(p0 - t + w)
                if num_begin < 1:
                    num_begin = 1
                if num_end > max_num:
                    num_end = max_num

                h = sample_curve_spl_ex[num_begin:num_end]

                e = np.arange(num_begin, num_end)
                index = np.round(xscale * (e + t) + (1 - xscale) * p0).astype(int)
                index[index < 0] = 0
                index[index >= max_num] = max_num - 1
                g_pre = shape_model_spl_ex[index]

                # R
                R[t + 30] = pearsonr(g_pre, h)[0]

            R_max = np.max(R)
            tshift_close = np.argmax(R) - 31
            del R, g_pre

            # break
            if q != 0 and tshift_close == tshift:
                if R_max > 0.8:
                    output[point] = p0 - tshift
                break

            # xscale
            tshift = tshift_close
            R = np.zeros(41)

            for x in np.arange(0.8, 1.21, 0.01):
                num_begin = (p0 - tshift - w).astype(int)
                num_end = (p0 - tshift + w).astype(int)
                if num_begin < 1:
                    num_begin = 1
                if num_end > max_num:
                    num_end = max_num

                h = sample_curve_spl_ex[num_begin:num_end]
                e = np.arange(num_begin, num_end)
                index = np.round(x * (e + tshift) + (1 - x) * p0).astype(int)
                index[index < 0] = 0
                index[index >= max_num] = max_num - 1
                g_pre = shape_model_spl_ex[index]

                # R
                R[np.round(x * 100 - 80).astype(int)] = pearsonr(g_pre, h)[0]

            R_max = np.max(R)
            xscale_close = np.argmax(R) + 79
            xscale = xscale_close / 100
            del R, g_pre

    return output


def calculate_rmse_mae(df1, df2):
    """
    Calculate the RMSE and MAE of the corresponding columns of the two DataFrames, ignoring the NaN values.

    Parameter:
    df1, df2: Pandas DataFrame
    The two DataFrames that need to be compared.
    """

    assert set(df1.columns) == set(df2.columns), "DataFrames must have the same columns"

    rmse_dict = {}
    mae_dict = {}

    for column in df1.columns:

        combined = pd.concat([df1[column], df2[column]], axis=1).dropna()
        y_true = combined.iloc[:, 0]
        y_pred = combined.iloc[:, 1]

        # compute RMSE and MAE
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)


        rmse_dict[column] = rmse
        mae_dict[column] = mae
        res_df = pd.DataFrame(columns=rmse_dict.keys())
        res_df = pd.concat([res_df, pd.DataFrame([rmse_dict])], ignore_index=True)
        res_df = pd.concat([res_df, pd.DataFrame([mae_dict])], ignore_index=True)
        res_df.index = ["RMSE", "MAE"]
    return res_df

# LOWESS
def apply_lowess_to_row(row, frac=0.2):
    x = np.arange(len(row))
    y = row.values
    lowess_smoothed = sm.nonparametric.lowess(y, x, frac=frac)[:, 1]
    return pd.Series(lowess_smoothed, index=row.index)


def find_min_rmse_df_and_row(df_list, target_array):
    target_array = target_array.ravel() 
    min_rmse = float('inf')
    result_df_idx = -1
    result_row_pos = -1

    for df_idx, df in enumerate(df_list):
        data = df.values  
        #compute RMSE
        squared_diff = (data - target_array) ** 2
        rmse_values = np.sqrt(squared_diff.mean(axis=1))

        current_min_pos = rmse_values.argmin()
        current_min_rmse = rmse_values[current_min_pos]
        if current_min_rmse < min_rmse:
            min_rmse = current_min_rmse
            result_df_idx = df_idx
            result_row_pos = current_min_pos  

    return result_row_pos


def smf(shape_model, sample_curve, shape_model_point):
    best_paramter_arr = np.array([np.nan, np.nan, np.nan, float('inf')])
    phenology_detect = np.zeros(shape=shape_model_point.shape)

    tshift_arr = np.arange(start=-30, step=2, stop=31)
    xscale_arr = np.arange(start=0.8, step=0.05, stop=1.21)
    yscale_arr = np.arange(start=0.9, step=0.05, stop=1.11)

    # tshift
    shift_res = shift_and_fill(pd.DataFrame(shape_model).T)
    paras = []
    # xscale
    xscale_shift_res = []
    for sigma in xscale_arr:
        for i, shift_df in enumerate(shift_res):
            if sigma == 1:
                xscale_shift_res.append(shift_df)
                paras.append([tshift_arr[i], sigma])
            else:
                time_stretch_arr = time_stretch(shift_df, sigma)
                xscale_shift_res.append(time_stretch_arr)
                paras.append([tshift_arr[i], sigma])
    shift_xscale_df = pd.concat(xscale_shift_res, ignore_index=True)
    # yscale
    final_res = []
    for yscale in yscale_arr:
        final_res.append(np.round(yscale, 2) * shift_xscale_df)

    # find best parameter
    xscale_shift_optimal_index = find_min_rmse_df_and_row(final_res, sample_curve)
    tshift_optimal, xscale_optimal = paras[xscale_shift_optimal_index]
    # 计算物候探测结果
    for i in range(shape_model_point.size):
        phenology_detect[i] = np.round(xscale_optimal * (shape_model_point[i] + tshift_optimal))

    return phenology_detect


def smf_detect(temporal_df, reference_curve_, reference_phenology):

    smf_res = pd.DataFrame(index=temporal_df.index, columns=reference_phenology.index)
    for index, row in temporal_df.iterrows():
        print(f"当前计算下标为{index}")
        smf_res.loc[index] = smf(reference_curve_.loc[0].values, row.values, reference_phenology.values)
    return smf_res


def smooth_crop_offseason_signal(series):
    s_diff = series.diff()
    max_value_idx = series.idxmax()

    found_left = found_right = False
    left_idx = right_idx = None

    # Look to the left for the nearest turning point
    for i in range(max_value_idx - 6, series.index[0], -6):
        if s_diff.loc[i] * s_diff.loc[max_value_idx - 6] <= 0:
            left_idx = i
            found_left = True
            break

    # Look to the right for the nearest turning point
    for i in range(max_value_idx + 6, series.index[-1], 6):
        if i in s_diff.index and s_diff.loc[i] * s_diff.loc[max_value_idx + 6] <= 0:
            right_idx = i
            found_right = True
            break

    # Returns results based on whether a turning point was found
    if not found_left or max_value_idx - left_idx <= 30:
        left_idx = 58
    if not found_right or right_idx - max_value_idx <= 30:
        right_idx = 334

    if max_value_idx <= 100 or max_value_idx >= 300:
        left_idx = 58
        right_idx = 334

    series.loc[:left_idx] = series.loc[left_idx]
    series.loc[right_idx:] = series.loc[right_idx]

    return series


# The mean is smooth, and the window length is 3
def moving_average_filter(row, window_size=3):

    half_window = window_size // 2
    filtered_row = row.copy()  

    for i in range(len(row)):
        if i < half_window or i >= len(row) - half_window:
            filtered_row.iloc[i] = row.iloc[i]
        else:
            filtered_row.iloc[i] = np.mean(row.iloc[i - half_window:i + half_window + 1])

    return filtered_row


def time_stretch(df, scale_factor):
    """
    Scales the values of each row of the DataFrame and calculates the median value by a specific range.

    Parameter:
    df (pandas. DataFrame): A DataFrame containing doy columns (i.e., the column name is doy values) and corresponding values
    scale_factor (float): Scale factor
    """

    x_new = df.columns * scale_factor

    new_data = []
    for idx, row in df.iterrows():

        f = interp1d(x_new, row.values, kind='linear', fill_value='extrapolate')
        y_new = f(df.columns)
        new_data.append(y_new)
    new_df = pd.DataFrame(new_data, columns=df.columns, index=df.index)
    return new_df


def calculate_gmfr(data_this_year, window_pixel_data):
    """
    Calculate GMFR results and MSD values.

    Parameter:
    data_this_year (numpy.ndarray): Data for the current year
    window_pixel_data (numpy.ndarray): Pixel data within the window
    """
    m1, b1 = np.polyfit(window_pixel_data, data_this_year, 1)
    m2, b2 = np.polyfit(data_this_year, window_pixel_data, 1)
    m = np.sqrt(m1 * m2)
    b = np.mean(data_this_year) - m * np.mean(window_pixel_data)
    gmfr_res = m * window_pixel_data + b
    msd = np.mean((data_this_year - gmfr_res) ** 2)
    return gmfr_res, msd


def gmfr(df_one_row, df_window):
    """
    Calculate GMFR for each window, returning the GMFR result for the window with the smallest MSD.

    Parameter:
    df_one_row (pandas. DataFrame): A DataFrame containing data for the current year
    df_window (pandas. DataFrame): The DataFrame of the window data
    """
    data_this_year = df_one_row.iloc[0].values
    msd_list = []

    for window_pixel_data in df_window.values:
        gmfr_res, msd = calculate_gmfr(data_this_year, window_pixel_data)
        msd_list.append(msd)

    min_msd_index = np.argmin(msd_list)
    best_window_pixel_data = df_window.iloc[min_msd_index].values

    # Calculate the GMFR results for the optimal window
    gmfr_res, _ = calculate_gmfr(data_this_year, best_window_pixel_data)

    return pd.Series(gmfr_res, index=df_window.columns)


# Time scale translation
def shift_and_fill(df):
    """
    Perform an offset in the column direction for a given DataFrame and process the NaN value using forward or 
    backward padding depending on the direction of the displacement.
    """
    eachWindow_df_filtered_shift_list = []
    for i in range(-30, 31, 2):
        eachWindowData_df_filtered_shifted = df.shift(i, axis=1)

        if i < 0:
            eachWindowData_df_filtered_shifted = eachWindowData_df_filtered_shifted.ffill(axis=1)
        elif i > 0:
            eachWindowData_df_filtered_shifted = eachWindowData_df_filtered_shifted.bfill(axis=1)
        eachWindow_df_filtered_shift_list.append(eachWindowData_df_filtered_shifted)

    return eachWindow_df_filtered_shift_list