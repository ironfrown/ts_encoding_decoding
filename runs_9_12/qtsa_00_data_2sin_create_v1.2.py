#!/usr/bin/env python
# coding: utf-8

# # Data Generation: Simple Time Series
# Compatible with Qiskit 1.2.4+

# ### Author
# - **Jacob Cybulski**, jacob.cybulski[at]deakin.edu.au<br/>
#     School of IT, SEBE, Deakin University, Melbourne, Vic, Australia
# 
# ### Date
# - October 2022: Prepared for Workshop on Quantum Machine Learning, 13 October 2022, organised in collaboration with QWorld, QPoland, QIndia and Quantum AI Foundation. In association with IEEE Conference Trends in Quantum Computing and Emerging Business Technologies - TQCEBT 2022
# 
# ### Aims
# > *This script aims to create and save a simple time series, useful for curve fitting (no windows).*
# 
# ### Note
# > *<font color="tomato">When running this script, you are likely to obtain slightly different results each time.</font>*

# In[1]:


import sys
sys.path.append('.')
sys.path


# In[2]:


import os
import numpy as np
import math
import argparse

from utils.Target import *
from utils.Charts import *
from utils.Files import *

import matplotlib.pyplot as plt
from matplotlib import set_loglevel
set_loglevel("error")

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ## Set the program constants

# In[3]:


### interactive control
try:
    __IPYTHON__
    interactive = True
    print(f'\nInteractive execution started\n')
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    interactive = False
    print(f'\nBatch execution started\n')
    
### Software version
MAJOR = 9
MINOR = 0


# ## Set program parameters
# *Defaults are set here*

# In[4]:


### Constants
LOG_NAME = 'log_test'
DATA_NAME = '2_sins'
FUN_NAME = 'Target_2_sins'

### Data settings
samples_train = 70
samples_valid = 30
noise = 0.0
wind_size = 0
wind_step = 0
horizon = 0
seed = 1410 # The Battle of Grunwald


# ## Prepare data
# *Values need to be in [0, 1] range*

# In[5]:


### Select a target class and its parameters (see: utils.py), e.g.
#   Target_sin() # Target_2_sins() # Target_poly() # Target_poly_3()
#   Target_line() # Target_line(slope=0.5, intercept=0.2, xmin=0, xmax=1.5)
#   Target_trig_trend() # Target_jitter()
#   Target_beer() # Target_beer(pt_from=104, pt_to=156)

### Function to be used
f = target_by_name(FUN_NAME)()

### Data paths within a log file
DATA_PATH = f'{LOG_NAME}/data'

### Calculated values and functions
samples = samples_train+samples_valid
train_pc = samples_train / samples


# ## Prepare a simple TS version of data
# *Values need to be in [0, 1] range*

# In[6]:


### Use a function f to create a data set and its partitions for training and validation

def ts_prep_data(f, samples=100, noise=0.0, train_pc=0.7, seed=1410):
    # Create data points
    X_all, y_all, X_train, y_train, X_valid, y_valid = \
        target_split(f, samples, noise=noise, train_pc=train_pc, seed=seed)
    
    return X_all, y_all, X_train_ts, y_train_ts, X_valid_ts, y_valid_ts

if interactive:
    X_all, y_all, X_train, y_train, X_valid, y_valid = \
        target_split(f, samples, noise=noise, train_pc=train_pc, seed=seed)
    print(f'\nCreated a data set "{f.name}": windows# train={len(y_train)}, valid={len(y_valid)}, noise={noise}\n')


# In[7]:


### All created files have the following codes:
#   t: training sample size
#   v: validation sample size
#   z: level of noise injected = 0
#   w: window size = 0
#   s: window step size = 0
#   h: horizon size = 0
def ts_data_id(data_name, samples_train, samples_valid, noise=0.0):
    return f'{data_name}_t{samples_train}_v{samples_valid}_z{noise}_w{0}_s{0}_h{0}'

if interactive:
    DATA_ID = ts_data_id(DATA_NAME, samples_train, samples_valid, noise=noise)
    print(f'\nUnique DATA_ID = "{DATA_ID}"\n')


# ## Test plot the data

# In[8]:


if interactive:

    ### Plot data
    plot_train_and_test_data(
        X_all, y_all, X_train, y_train, X_valid, y_valid,
        xlim=(-6.7, 6.7), ylim=(0.05, 1),
        colors=['lightblue', 'blue', 'red', 'blue', 'red'], linestyles=['dashed', 'solid', 'solid'],
        title=f'Dataset for curve fitting ("{DATA_NAME}" data, with {samples} samples, {samples_train}-{samples_valid} split)',
    )


# ## Save created data

# In[9]:


### Save a data set
def ts_save_dataset(dpath, did,
                    X_all, y_all, X_train, y_train, X_valid, y_valid):

    ### Define file paths
    x_all_fpath = f'{dpath}/{did}/x_all.arr'
    y_all_fpath = f'{dpath}/{did}/y_all.arr'
    
    x_train_fpath = f'{dpath}/{did}/x_train.arr'
    y_train_fpath = f'{dpath}/{did}/y_train.arr'
    x_valid_fpath = f'{dpath}/{did}/x_valid.arr'
    y_valid_fpath = f'{dpath}/{did}/y_valid.arr'

    ### Save a data set and its partitions
    write_ts_file(x_all_fpath, X_all)
    write_ts_file(y_all_fpath, y_all)
    write_ts_file(x_train_fpath, X_train)
    write_ts_file(y_train_fpath, y_train)
    write_ts_file(x_valid_fpath, X_valid)
    write_ts_file(y_valid_fpath, y_valid)

if interactive:
    ts_save_dataset(DATA_PATH, DATA_ID,
                    X_all, y_all, X_train, y_train, X_valid, y_valid)
    print(f'\nSaved time series data in: "{DATA_PATH}/{DATA_ID}"\n')


# In[10]:


### Save info details
def ts_save_dataset_info(dpath, did, data_name,
    samples_train, samples_valid, noise, seed=2024):
    global MAJOR, MINOR
    
    ### Define an info path
    data_info_fpath = f'{dpath}/{did}/info.json'
    
    ### Save data info details
    data_info = \
        {'data_name':data_name, 
         'major_version':MAJOR, 
         'minor_version':MINOR,
         'data_train':samples_train,
         'data_valid':samples_valid,
         'data_noise':noise,
         'wind_size':0,
         'wind_step':0,
         'wind_horizon':0,
         'seed':seed}
    
    ### Saving the info file
    write_json_file(data_info_fpath, data_info)

    return data_info, data_info_fpath

if interactive:
    data_info, info_fpath = ts_save_dataset_info(DATA_PATH, DATA_ID, DATA_NAME,
        samples_train, samples_valid, noise, seed=seed)
    print(f'\nSaved time series info in file "{info_fpath}":\n')
    for k in data_info.keys():
        print(f'\tinfo[{k}] = {data_info[k]}')
    print()


# ## System details

# In[11]:


import sys
print(f'Environment:\n\n{sys.prefix}\n')


# In[12]:


import os
print(f"Significant packages:\n")
os.system('pip list | grep -e qiskit -e torch');


# ## Main

# In[13]:


### Fetch all program parameters
#   log: Log name
#   -d --data: Data name
#   -f --fun: Function names
#   -sn --samples: sample size
#   -st --tsamples: training samples
#   -sv --vsamples: validation samples
#   -z: level of noise injected
#   -rs --seed: random seed

def sw_get_vars():
    global LOG_NAME, DATA_NAME, FUN_NAME, DATA_PATH
    global samples, samples_train, samples_valid, train_pc, noise
    global wind_size, wind_step, horizon
    global f, seed

    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=str, help="log relative path")
    parser.add_argument("-d", "--data", type=str, help="data name", default="sin")
    parser.add_argument("-f", "--fun", type=str, help="name of generating function", default="Target_sin")
    parser.add_argument("-sn", "--samples", type=int, help="sample size", default=100)
    parser.add_argument("-st", "--tsamples", type=int, help="size of training sample", default=70)
    parser.add_argument("-sv", "--vsamples", type=int, help="size of validation sample", default=30)
    parser.add_argument("-pcs", "--split", type=float, help="percentage split", default=0.7)
    parser.add_argument("-z", "--noise", type=float, help="noise level", default=0.0)
    parser.add_argument("-rs", "--seed", type=int, help="random seed", default=2024)

    args = parser.parse_args()
    
    return args


# In[14]:


### Main function, executed in batch mode only

def main():
    global LOG_NAME, DATA_NAME, FUN_NAME, DATA_PATH
    global samples, samples_train, samples_valid, train_pc, noise
    global wind_size, wind_step, horizon
    global f, seed

    ### Execute the process
    print(f'\nSteps taken:')

    ### Create a data set
    X_all, y_all, X_train, y_train, X_valid, y_valid = \
        target_split(f, samples, noise=noise, train_pc=train_pc, seed=seed)
    print(f'\n1. Created a data set "{f.name}": windows# train={len(y_train)}, valid={len(y_valid)}, noise={noise}\n')

    ### Create a unique data ID
    DATA_ID = ts_data_id(DATA_NAME, samples_train, samples_valid, noise=noise)
    print(f'\n2. Unique DATA_ID = "{DATA_ID}"\n')

    ### Save a dataset
    ts_save_dataset(DATA_PATH, DATA_ID,
        X_all, y_all, X_train, y_train, X_valid, y_valid)
    print(f'\n3. Saved time series data in: "{DATA_PATH}/{DATA_ID}"\n')

    ### Save dataset info
    data_info, info_fpath = ts_save_dataset_info(DATA_PATH, DATA_ID, DATA_NAME,
        samples_train, samples_valid, noise, seed=seed)
    print(f'\n4. Saved time series info in file "{info_fpath}":\n')
    for k in data_info.keys():
        print(f'\tinfo[{k}] = {data_info[k]}')
    print()


# In[15]:


### Execute the process
if not interactive:

    ## Get external variables
    args = sw_get_vars()

    ## Assign locals vars
    LOG_NAME = args.log
    DATA_NAME = args.data
    FUN_NAME = args.fun
    DATA_PATH = f'{LOG_NAME}/data'

    ## Get the windows parameters
    samples_train = args.tsamples
    samples_valid = args.vsamples
    noise = args.noise
    wind_size = 0
    wind_step = 0
    horizon = 0
    seed = args.seed

    ## Calculate important vars
    samples = samples_train+samples_valid
    train_pc = samples_train / samples
    f = target_by_name(FUN_NAME)()

    main()

