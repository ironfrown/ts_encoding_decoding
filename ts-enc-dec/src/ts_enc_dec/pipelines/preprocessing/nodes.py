"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.19.10
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from .target import target_by_name


def simple_ts_data_prep(f, params):
    # Create data points
    f = target_by_name(f)()
    x_all, y_all, x_train, y_train, x_valid, y_valid = _target_split(f, **params)

    x_all = pd.DataFrame(x_all,columns=['data'])
    y_all = pd.DataFrame(y_all,columns=['data'])
    x_train = pd.DataFrame(x_train,columns=['data'])
    x_valid = pd.DataFrame(x_valid,columns=['data'])
    y_train = pd.DataFrame(y_train,columns=['data'])
    y_valid = pd.DataFrame(y_valid,columns=['data'])

    return dict(
        x_all=x_all,
        y_all=y_all,
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
    )

def _target_split(f, samples, noise=0, train_pc=0.7, seed=None):
    print(samples, noise, train_pc, seed)
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

# =========== PLOTS ==================

def plot_simple_data(X_all, y_all, X_train, y_train, X_valid, y_valid, params):

    plot = _plot_train_and_test_data(X_all.values, y_all.values, X_train.values, y_train.values,
                                    X_valid.values, y_valid.values, xlim=tuple(params['xlim']), 
                                    ylim=tuple(params['ylim']), colors=params['colors'], 
                                    linestyles=params['linestyles'], title=params['title'])

    return plot


def _plot_train_and_test_data(
    X_org, y_org, X_train, y_train, X_valid, y_valid,
    y_train_hat=None, y_valid_hat=None,
    xlim=None, ylim=None, rcParams=(12, 6), dpi=72,
    legend_cols=3,
    labels=['Target function', 'Training data', 'Test data', 'Fitted model', 'Model predictions'],
    colors=['lightblue', 'lightblue', 'pink', 'blue', 'red'],
    linestyles=['dashed', 'solid', 'solid', 'dashed', 'dashed'],
    xlabel='Range', ylabel='Target value',
    title='Target function with noisy data'):

    # Parameter values
    if rcParams is not None:
        plt.rcParams["figure.figsize"] = rcParams
    plt.rcParams["figure.dpi"] = dpi
        
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Plot target function
    if linestyles[0] != 'none':
        plt.plot(X_org, y_org, color=colors[0], linestyle=linestyles[0], label=labels[0])
    plt.plot(X_train, y_train, color=colors[1], linestyle=linestyles[1], label=labels[1])
    plt.plot(X_valid, y_valid, color=colors[2], linestyle=linestyles[2], label=labels[2])
    
    # Plot fitted line
    if y_train_hat is not None:
        plt.plot(X_train, y_train_hat, color=colors[3], linestyle=linestyles[3], label=labels[3])
        plt.plot(X_train, y_train_hat, color=colors[3], marker='o', linestyle='None')
    else:
        plt.plot(X_train, y_train, color=colors[3], marker='o', linestyle='None')

    # Plot prediction
    if y_valid_hat is not None:
        plt.plot(X_valid, y_valid_hat, color=colors[4], linestyle=linestyles[4], label=labels[4])
        plt.plot(X_valid, y_valid_hat, color=colors[4], marker='o', linestyle='None')
    else:
        plt.plot(X_valid, y_valid, color=colors[4], marker='o', linestyle='None')

    plt.axvline(x = (X_train[-1][0]+X_valid[0][0])/2, color = 'lightgray', linestyle='dashed')
    plt.legend(loc='best', ncol=legend_cols)
    
    return plt  