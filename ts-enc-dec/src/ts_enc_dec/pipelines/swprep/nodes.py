"""
Sliding windows
generated using Kedro 0.19.10
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from .target import target_by_name
    
def sw_prep_data(f, params):
    # Create data points
    f = target_by_name(f)()

    X_all, y_all, _, _, _, _ = \
        _target_split(f, params['samples'], noise=params['noise'], train_pc=1.0, seed=params['seed'])
    
    # Prepare X and y for training
    X_train_ts, y_train_ts, X_valid_ts, y_valid_ts = \
        _Xy_wind_split(y_all, params['wind_size'], params['wind_step'], params['horizon'], params['train_pc'])
    
    col_names = [f"data_{i+1}" for i in range(params['wind_size'])]


    X_all_sw = pd.DataFrame(X_all, columns=['data'])
    y_all_sw = pd.DataFrame(y_all, columns=['data'])
    x_train_sw = pd.DataFrame(X_train_ts, columns=col_names)
    x_valid_sw = pd.DataFrame(X_valid_ts, columns=col_names)
    y_train_sw = pd.DataFrame(y_train_ts, columns=['data'])
    y_valid_sw = pd.DataFrame(y_valid_ts, columns=['data'])

    return X_all_sw, y_all_sw, x_train_sw, y_train_sw, x_valid_sw, y_valid_sw


def _target_split(f, samples, noise=0, train_pc=0.7, seed=None):
    if seed is not None: np.random.seed(seed)
    samples_train = int(samples * train_pc)
    samples_valid = samples-samples_train
        
    lb, ub = f.x_range()
    lb_train, ub_train = lb, lb+train_pc*(ub - lb)
    lb_valid, ub_valid = lb+train_pc*(ub - lb), ub
    T = (ub - lb)
    
    ### Prepare all X and y data
    X_all = np.linspace(lb, ub, num=samples)
    y_all = f.fun(X_all)
    
    ### Some of these are legacy
    X_train = (ub_train - lb_train) * np.random.random(samples_train) + lb_train
    X_train = np.sort(X_train, axis = 0)
    y_train = f.fun(X_train) + noise * (np.random.random(samples_train) - 0.5)
    X_valid = (ub_valid - lb_valid) * np.random.random(samples_valid) + lb_valid
    X_valid = np.sort(X_valid, axis = 0)
    y_valid = f.fun(X_valid) + noise * (np.random.random(samples_valid) - 0.5)
    
    ### Reshape Xs for fitting, scoring and prediction
    X_all = X_all.reshape(samples, 1)
    X_train = X_train.reshape(samples_train, 1)
    X_valid = X_valid.reshape(samples_valid, 1)

    return X_all, y_all, X_train, y_train, X_valid, y_valid

def _Xy_wind_split(y, wind, step, horizon, split):
    X, y = _Xy_wind_make(y, wind, step, horizon)
    return _ts_wind_split(X, y, split)

def _Xy_wind_make(y, wind, step, horizon):
    full_wind = wind + horizon
    Xy_wind = _y_wind_make(y, full_wind, step)
    return Xy_wind[:,:wind], Xy_wind[:,wind:]

def _ts_wind_split(X, y, split):
    train_size = int(np.round(X.shape[0] * split, 0))
    return X[:train_size], y[:train_size], X[train_size:], y[train_size:]

def _y_wind_make(y, wind, step):
    return _ts_wind_make(y, wind, step)

def _ts_wind_make(ts, wind, step):
    ts_wind = np.array([np.array(ts[i:i+wind]) for i in range(0, len(ts)-wind+1, step)])
    return ts_wind

# ================= PLOTS ==============


def plot_flat(y_all_sw, y_train_sw, y_valid_sw, dparam, params):
    
    y_all = y_all_sw.values
    y_train_ts = y_train_sw.values
    y_valid_ts = y_valid_sw.values


    y_all_select = y_all[dparam['wind_size']:]
    y_train_flat_ts = _ts_wind_flatten_avg(y_train_ts, dparam['wind_step'])
    y_valid_flat_ts = _ts_wind_flatten_avg(y_valid_ts, dparam['wind_step'])
    y_list = [y_all_select, y_train_flat_ts, y_valid_flat_ts]
    X_list = [0, 0, len(y_train_flat_ts)]

    plot = _multi_plot_flat_ts(y_list=y_list, X_list=X_list, colors=params['colors'], 
                              marker_colors=params['marker_colors'],labels=params['labels'], 
                              ylim=tuple(params['ylim']), xlabel=params['xlabel'], ylabel=params['ylabel'],
                              lines=params['lines'],  markers=params['markers'],
                              legend_cols=params['legend_cols'], title=params['title'])

    return plot

def _ts_wind_flatten_avg(wind, wind_step):
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

def _multi_plot_flat_ts(
    y_list, X_list=None, labels=None, colors=None, lines=None, markers=None, marker_colors=None,
    xlim=None, ylim=None, rcParams=(12, 6), xlabel='Range', ylabel='Target value',
    legend_cols=3, title='Time series plot', save_plot=None):

    if len(y_list) == 0:
        print(f'*** Error: the list of data to plot cannot be empty')
        return
        
    # Missing main parameters
    if X_list is None or len(X_list) == 0:
        X_list = [0] # [[i for i in ]]
    if len(X_list) < len(y_list):
        ts_start = X_list[-1]+len(y_list[len(X_list)])
        for ts in y_list[len(X_list):]:
            X_list.append(ts_start)
            ts_start += len(ts)

    if labels is None or len(labels) == 0:
        labels = [f'Plot {0:02d}']
    if len(labels) < len(y_list):
        for i in range(len(labels), len(y_list)):
            labels.append(f'Plot {i:02d}')

    cmap = matplotlib.colormaps['Set1']
    map_colors = cmap.colors+cmap.colors+cmap.colors
    if colors is None or len(colors) == 0:
        colors = [map_colors[0]]
    if len(colors) < len(y_list):
        for i in range(len(colors), len(y_list)):
            colors.append(map_colors[i])

    if marker_colors is None or len(marker_colors) == 0:
        marker_colors = colors
    if len(marker_colors) < len(y_list):
        for i in range(len(marker_colors), len(y_list)):
            marker_colors.append(colors[i])

    if lines is None or len(lines) == 0:
        lines = ['solid']
    if len(lines) < len(y_list):
        lines = lines+['solid']*(len(y_list)-len(lines))
    
    if markers is None or len(markers) == 0:
        markers = ['none']
    if len(markers) < len(y_list):
        markers = markers+['none']*(len(y_list)-len(markers))
    
    # Parameter values
    if rcParams is not None:
        plt.rcParams["figure.figsize"] = rcParams        
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Calculate the X_range for each list
    
    for i in range(len(y_list)):
        xfirst = X_list[i]
        xlast = X_list[i]+len(y_list[i])
        xrange = range(xfirst, xlast)
        plt.plot(xrange, y_list[i], linestyle=lines[i], marker=markers[i], 
                 color=colors[i], mec=colors[i], mfc=marker_colors[i], label=labels[i])
        if i > 0:
            plt.axvline(x = X_list[i]-0.5, color = 'lightgray', linestyle='dashed')
    
    plt.legend(loc='best', ncol=legend_cols)
    
    return plt  