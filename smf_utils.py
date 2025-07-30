import numpy as np
from numba import jit
import time



def batch_shift_and_pad(array, tshifts):
    # Generate all the translated vectors
    shape = (tshifts.size, array.shape[0])
    res = np.zeros(shape, dtype=np.float64)
    for i, t in enumerate(tshifts):
        res[i] = shift_and_pad(array, t).astype(np.float64)
    return res



def shift_and_pad(array, tshift):
    if tshift == 0:
        return array.copy()

    fill_value = array[0] if tshift > 0 else array[-1]
    result = np.full_like(array, fill_value)
    # Adjust the data position according to the translation direction
    if tshift > 0:
        result[tshift:] = array[:-tshift]  
    else:
        abs_shift = -tshift
        result[:-abs_shift] = array[abs_shift:]  
    return result


#Data interpolation
def stretch_interpolation(data, scale):
    original_x = np.arange(1, 362)  
    scaled_x = original_x * scale  
    target_x = original_x.copy() 


    interpolated_data = np.zeros_like(data)

    for row in range(data.shape[0]):
        y_values = data[row, :]  
        interpolated_y = np.interp(
            target_x,
            scaled_x,
            y_values
        )

        mask_left = target_x < scaled_x[0]  
        mask_right = target_x > scaled_x[-1]  

        interpolated_y[mask_left] = y_values[0]
        interpolated_y[mask_right] = y_values[-1]
        interpolated_data[row, :] = interpolated_y

    return interpolated_data


#Batch interpolation
def batch_stretch_interpolation(array, xscales):

    shape = (xscales.size, array.shape[0], array.shape[1])
    res = np.zeros(shape, dtype=np.float64)
    for i, x in enumerate(xscales):
        res[i] = stretch_interpolation(array, x).astype(np.float64)
    return res


def unravel_index(index, shape):

    coords = []
    num_dims = len(shape)
    for i in range(num_dims - 1, -1, -1):  
        coords.append(index % shape[i])
        index //= shape[i]
    return coords[::-1]


def find_min_rmse_index(A, B):
    """
    Calculate the RMSE of the last dimensional array of A and B, 
    and return the dimensional coordinates corresponding to the minimum RMSE
    """

    diffs = B - A.reshape(1, 1, 1, 361)  
    squared_errors = diffs ** 2
    mse = np.sum(squared_errors, 3) / squared_errors.shape[3]
    rmse = np.sqrt(mse)

    min_indices = np.unravel_index(np.argmin(rmse), rmse.shape)
    return min_indices


def smf_each_series(series, reference, phenology_date):
    tshift_arr = np.arange(start=-30, step=3, stop=31)
    xscale_arr = np.arange(start=0.8, step=0.05, stop=1.21)
    yscale_arr = np.arange(start=0.9, step=0.05, stop=1.11)

    # tshift 
    tshift_res = batch_shift_and_pad(reference, tshift_arr)

    # xscale
    tshift_scale_res = batch_stretch_interpolation(tshift_res, xscale_arr)

    # yscale
    shape = (yscale_arr.size, tshift_scale_res.shape[0], tshift_scale_res.shape[1], tshift_scale_res.shape[2])
    res = np.zeros(shape, dtype=np.float64)
    for i, y in enumerate(yscale_arr):
        res[i] = y * tshift_scale_res

    # compute Rmse
    indexes = find_min_rmse_index(series, res)
    xscale_optimal = xscale_arr[indexes[1]]
    tshift_optimal = tshift_arr[indexes[2]]

    return np.round(xscale_optimal * (phenology_date + tshift_optimal))


def smf(temporals, reference, phenology_dates):
    smf_res = np.zeros((temporals.shape[0], phenology_dates.size))
    for i in range(temporals.shape[0]):
        smf_res[i] = smf_each_series(temporals[i], reference, phenology_dates)
    return smf_res
