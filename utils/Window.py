# Support functions for QTSA workshop
# Author: Jacob Cybulski, ironfrown[at]gmail.com
# Aims: Provide support for quantum time series analysis
# Date: 2023

import matplotlib.pyplot as plt
import numpy as np
import pylab
import math
from typing import Union
from IPython.display import clear_output


##### Reshape data to allow windowing
#     It leaves a window size gap between training and validation partition
#     - To be fixed in the future

### Converts a flat time series to a windowed set of records
#   ts: series of values
#   wind: specific window size
#   step: step between windows
#   returns: a set of sliding windows of ts
def ts_wind_make(ts, wind, step):
    ts_wind = np.array([np.array(ts[i:i+wind]) for i in range(0, len(ts)-wind+1, step)])
    return ts_wind

### Converts a windowed set of records into a flat time series
#   ts: a windowed series of values
#   step: step between windows, to make sense it assumes step <= window size
#   returns: a flat time series
def ts_wind_flatten(ts, step):
    if ts.size == 0:
        return ts
    elif ts.shape[1] < step:
        flat_list = []
        for w in ts: flat_list += list(w)
        return np.array(flat_list)
    else:
        flat_list = []
        for w in ts: flat_list += list(w[0:step])
        flat_list += list(ts[-1][step:])
        return np.array(flat_list)

### Converts a windowed set of records into a flat time series by averaging overlapping windows
#     If step > window size, it will fill in the missing values with zeros
#   ts: a windowed series of values
#   step: step between windows
#   returns: a flat averaged time series
def ts_wind_flatten_avg(wind, wind_step):
    n_wind = wind.shape[0]
    if n_wind == 0: 
        return np.array(wind)
    else:
        # Collect windows sums and counts
        wind_size = wind.shape[1]
        n_elems = wind_step * n_wind + (wind_size - wind_step)
        
        wind_avg = np.zeros((n_elems), dtype=float)
        wind_sums = np.zeros((n_elems), dtype=float)
        wind_count = np.zeros((n_elems), dtype=int)
        
        for wind_no in range(n_wind):
            #print(f'Within window: {wind_no}')
            for wind_i in range(wind_size):
                wind_left_edge = wind_no * wind_step
                elem_i = wind_left_edge + wind_i
                # print(f'\ti={i} of {n_elems}, w_no={w_no} of {start_wind_no}, w[{wind_offset}]({wind_left_edge}, {wind_right_edge})')
                wind_sums[elem_i] += wind[wind_no][wind_i]
                wind_count[elem_i] += 1

        # Average overlapping windows
        for i in range(n_elems):
            if wind_count[i] == 0:
                wind_avg[i] == 0
            else:
                wind_avg[i] = wind_sums[i] / wind_count[i]

        return wind_avg
        
### Splits windowed time series (X, y) into training and validation parts
#   X, y: time series split into sliding windows
#   split: percentage of data to be used for training, the rest for validation
#   returns: sliding windows of X and y split into training and validation partitions
def ts_wind_split(X, y, split):
    train_size = int(np.round(X.shape[0] * split, 0))
    return X[:train_size], y[:train_size], X[train_size:], y[train_size:]


##### The following functions deal split a time series
#     so that the window is used as predictor and horizon as target

### Converts a flat time series to a windowed set of records
#   - Ignores X coordinates, so points are assumed equidistant
#   y: time series, where indeces are equidistant
#   wind: specific window size
#   step: step between windows
#   returns: a set of sliding windows of y
def y_wind_make(y, wind, step):
    return ts_wind_make(y, wind, step)

### Converts a flat time series into X and y set of records
#   y: time series, where indeces are equidistant
#   wind: specific window size
#   step: step between windows
#   horizon: the number of future data points to be predicted and used as y, if 0 no prediction
#   returns: sliding windows of X and y
def Xy_wind_make(y, wind, step, horizon):
    full_wind = wind + horizon
    Xy_wind = y_wind_make(y, full_wind, step)
    return Xy_wind[:,:wind], Xy_wind[:,wind:]

### Splits windowed data into training and validation sets
#   y: time series, where indeces are equidistant
#   wind: specific window size
#   step: step between windows
#   horizon: the number of future data points to be predicted and used as y, if 0 no prediction
#   split: percentage of data to be used for training, the rest for validation
#   returns: sliding windows of X and y split into training X, Y and validation X, y
def Xy_wind_split(y, wind, step, horizon, split):
    X, y = Xy_wind_make(y, wind, step, horizon)
    # train_size = int(np.round(X.shape[0] * split, 0))
    # return X[:train_size], y[:train_size], X[train_size:], y[train_size:]
    return ts_wind_split(X, y, split)
