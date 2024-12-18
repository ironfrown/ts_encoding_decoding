# Support functions for TS metrics
# Author: Jacob Cybulski, ironfrown[at]gmail.com
# Aims: Provide support for quantum time series analysis
# Date: 2024

### Cost function
#   This cost function can support several different cost types (and observables).
#   As the model is being optimised the cost plot will be produced continuously.
#   If model testing is initialised, the training and validation MAE are also calculated and plotted.
#   Model training with simultaneous testing is slow!
#
#   Note:
#   - As all intermediate parameters will be saved, the performance metrics can be calculated later.
#
#   To do:
#   - Modify to allow cost+parameters to be saved at some intervals only.

import sys
sys.path.append('.')
sys.path.append('..')
sys.path

import os
import numpy as np
import math
import random
import time
import datetime
import warnings
import matplotlib.pyplot as plt
from IPython.display import clear_output


##### Utilities

### Converts numbers to their binary representation
def digit_list(a, n_bits):
    return np.array([int(i) for i in f'{a:0{n_bits}b}'])

### Calculates an inner join on binary representation
def inner_join(a, b, n_bits):
    inna = digit_list(a, n_bits)
    innb = digit_list(b, n_bits)
    return np.inner(inna, innb)

### Makes an array indexed by number and 
#   returns the number of binary ones in its representation
def make_1s_count(bin_digs):
    dig_arr = [0]*bin_digs
    for i in range(bin_digs):
        dig_arr[i] = sum(digit_list(i, bin_digs))
    return np.array(dig_arr)


##### Manipulation of measurement probabilities

### Takes a list of probabilities (add to 1) and presents them in tabular form
def report_c2p(probs):
    qubit_no = int(np.log2(len(probs)))
    qa = np.array([[int(bit) for bit in format(qn, f'#0{qubit_no+2}b')[2:]] for qn in range(len(probs))])
    qr = np.zeros((qa.shape[0],qa.shape[1]+1))
    qr[:,:-1] = qa
    qr[:,qa.shape[1]:qa.shape[1]+1] = np.array([probs]).transpose()
    return qr

### Converts results of circuit state_vector measurements to 
#   individual qubit measurements
#   probs: a list of circuit measurement probabilities (must be of length 2^n)
#   returns: all qubit Z projective measurements onto |0>
def cprobs_to_qprobs(probs):
    qubit_no = int(np.log2(len(probs)))
    qpos = [[int(bit) for bit in format(qn, f'#0{qubit_no+2}b')[2:]] for qn in range(len(probs))]
    # qarr = np.array(qpos)
    qprobs = np.zeros(qubit_no)
    for qp in range(len(qpos)):
        for q in range(qubit_no):
            qprobs[q] += qpos[qp][q]*probs[qp]
    return qprobs

### Converts results of circuit state vector measurements to qubit angular positions
#   probs: a list of circuit measurement probabilities (must be of length 2^n)
#   returns 0: a list of qubit angular positions
#           1: a list of individual probabilities of qubit cast to |0> (towards positive values)
def cprobs_to_qangles(probs):
    qubit_0_probs = cprobs_to_qprobs(probs)
    qubit_0_probs = [1-x for x in qubit_0_probs] # compatibility with single_qubit_angle_meas

    all_q_meas = []
    for q in range(len(qubit_0_probs)):
        p0 = qubit_0_probs[q]
        p1 = 1 - p0
        amp0 = np.sqrt(p0)
        amp1 = np.sqrt(p1)
        meas_angle = 2*np.arccos(amp0)-np.pi/2
        all_q_meas.append(meas_angle)
    return all_q_meas, qubit_0_probs

### Single qubit measurement in terms of its angular position
#   Recall the figure explaining encoding
#   *** Should use encoding / decoding from Angles

from qiskit_aer.backends import AerSimulator
from qiskit_aer import Aer

def single_qubit_angle_meas(qc, backend, shots=10000):
    job = backend.run(qc, shots=shots)
    result = job.result()
    counts = result.get_counts(qc)
    
    counts0 = counts['0'] if '0' in counts.keys() else 0
    counts1 = counts['1'] if '1' in counts.keys() else 0
    p0 = counts0/(counts0+counts1)
    p1 = counts1/(counts0+counts1)
    amp0 = np.sqrt(p0)
    amp1 = np.sqrt(p1)

    meas_angle = 2*np.arccos(amp0)-np.pi/2
    return meas_angle, p0

### Multi-quibit measurement of individual qubits in terms of their angular position
#   Assumes a statevector backend and the state saved in the circuit
#   Recall the figure explaining encoding
#   qc: Circuit to be measured
#   backend: Backend to be used for measuring the circuit
#   shots (optional): The number of shots in execution
#   returns 0: a list of qubit angular positions
#           1: a list of individual probabilities of qubit cast to |0> (towards positive values)
#   *** Should use encoding / decoding from Angles

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.quantum_info import Statevector

def multi_qubit_angle_meas(qc, backend, shots=10000):
    result = backend.run(qc, shots=shots).result()
    qc_state = result.get_statevector()
    probs = Statevector(qc_state).reverse_qargs().probabilities(qargs=range(qc.num_qubits))
    qubit_0_probs = cprobs_to_qprobs(probs)
    qubit_0_probs = [1-x for x in qubit_0_probs] # compatibility with single_qubit_angle_meas

    all_q_meas = []
    for q in range(len(qubit_0_probs)):
        p0 = qubit_0_probs[q]
        p1 = 1 - p0
        amp0 = np.sqrt(p0)
        amp1 = np.sqrt(p1)
        meas_angle = 2*np.arccos(amp0)-np.pi/2
        all_q_meas.append(meas_angle)
    return all_q_meas, qubit_0_probs


##### Objective functions

### Calculates cost which is the probability of all measurements to be zero (measurement of 1)
#   If the repeated measurement of SWAP test returns 1 then the two states are identical (or highly similar). 
#   If the repeated measurement returns 0.5 then the two states are orthogonal (or highly dissimilar).
def cost_swap(pvals, probs):
    recs = probs.shape[0]
    return np.sum(probs[:, 1]) / recs

### Calculates cost which is 1 - the probability of measuring all qubits to be zero (|0>^n)
def cost_zero(pvals, probs):
    recs = probs.shape[0]
    return 1.0 - np.sum(probs[:, 0]) / recs

### Calculates cost which is 1 - the probability of measuring all qubits to be zero (|0>^n)
def cost_neg_zero(pvals, probs):
    recs = probs.shape[0]
    return -(1.0 - np.sum(probs[:, 0]) / recs)

### Calculates cost which is 1 - the sum of log(probability) of measuring qubits to be zero (|0>^n)
def cost_zero_log(pvals, probs):
    small_val = 0.0001
    recs = probs.shape[0]
    return -np.sum([(np.log(x) if x > small_val else small_val) for x in probs[:, 0]]) / recs

### Calculates the cost which is the weighted number of 1s per measurement
def cost_min1s(pvals, probs):
    digit_no = int(np.log2(probs.shape[1]+1))
    recs = probs.shape[0]
    digits_wsum = 0
    for rec in range(recs):
        for i in range(digit_no):
            digits_wsum += sum(digit_list(i, digit_no))*probs[rec, i]
    return digits_wsum


##### Higher-level objective functions (compare expected output with the predicted as calculated from probabilities)

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

### Collect all expected and predicted values
def collect_objfun_vals(expected_vals, probs):
    pred_vals = []
    exp_vals = []
    vals_no = 0
    for i in range(len(probs)):
        pred_vals += list(cprobs_to_qangles(probs[i])[0])
        exp_vals += list(expected_vals[i])
        vals_no += len(expected_vals[i])
    return np.array(exp_vals), np.array(pred_vals), vals_no

### Find the score between expected values and predicted QAE values as calculated from the results probabilities

def obj_fun_mae(expected_vals, probs):
    exp_vals, pred_vals, vals_no = collect_objfun_vals(expected_vals, probs)
    return mean_absolute_error(exp_vals, pred_vals)

def obj_fun_mse(expected_vals, probs):
    exp_vals, pred_vals, vals_no = collect_objfun_vals(expected_vals, probs)
    return mean_squared_error(exp_vals, pred_vals)

def obj_fun_rmse(expected_vals, probs):
    exp_vals, pred_vals, vals_no = collect_objfun_vals(expected_vals, probs)
    return np.sqrt(mean_squared_error(exp_vals, pred_vals))

def obj_fun_r2(expected_vals, probs):
    exp_vals, pred_vals, vals_no = collect_objfun_vals(expected_vals, probs)
    return 1-r2_score(exp_vals, pred_vals)



# make the plot nicer
plt.rcParams["figure.figsize"] = (12, 6) 


##### Cost class Regressor for NeuralNetworkRegressor

from qiskit_algorithms.utils import algorithm_globals

class Regr_callback:
    name = "Regr_callback"
    
    # Initialises the callback
    def __init__(self, log_interval=1, prompt_interval=1, tqdm_progress=None):
        self.objfun_min = 99999
        self.log_min = 99999
        self.objfun_vals = []
        self.params_vals = []
        self.epoch = 0
        self.log_interval = log_interval
        self.prompt_interval = prompt_interval
        self.pbar = tqdm_progress

    # Initialise callback lists
    # - For some reason [] defaults not always work (bug?)
    def reset(self):
        self.objfun_vals = []
        self.params_vals = []
        self.epoch = 0

    # Find the first minimum objective fun value
    def min_obj(self):
        if self.objfun_vals == []:
            return (-1, 0)
        else:
            minval = min(self.objfun_vals)
            minvals = [(i, v) for i, v in enumerate(self.objfun_vals) if v == minval]
            return minvals[0]

    # Creates a simple plot of the objective functionm
    # - Can be used iteratively to make animated plot
    def plot(self, title=None, xlabel=None, ylabel=None, col=None, save_plot=None, show_plot=True):
        clear_output(wait=True)
        if title is None: title = 'Objective function value'
        if xlabel is None: xlabel = 'Iteration'
        if ylabel is None: ylabel = 'Cost'
        if col is None: col = 'blue'
        best_val = self.min_obj()
        x_vals = [x*self.log_interval for x in range(len(self.objfun_vals))]
        plt.rcParams["figure.figsize"] = (12, 6)
        plt.title(f'{title} (min: {np.round(best_val[1], 4)} @ {best_val[0]*self.log_interval})')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(x_vals, self.objfun_vals, color=col)
        
        if save_plot is not None:
            os.makedirs(os.path.dirname(save_plot), exist_ok=True)
            plt.savefig(save_plot, format='eps')
        if show_plot:
            plt.show()

    def collect(self, weights, obj_func_eval):
        if self.objfun_min > obj_func_eval:
            self.objfun_min = obj_func_eval
        if self.epoch % self.log_interval == 0:
            self.objfun_vals.append(obj_func_eval)
            self.params_vals.append(weights)
            self.log_min = np.min(self.objfun_vals)
        if self.pbar is not None: self.pbar.update(1)
        self.epoch += 1

    # Callback function to store objective function values and plot
    def objfun_graph(self, weights, obj_func_eval):
        self.collect(weights, obj_func_eval)
        if self.epoch % self.prompt_interval == 0:
            self.plot()
            
    # Callback function to store objective function values but not plot
    def objfun_print(self, weights, obj_func_eval):
        self.collect(weights, obj_func_eval)
        if self.epoch % self.prompt_interval == 0:
            best_val = self.min_obj()
            print(f'Results:{"":<3}epoch={self.epoch: 5d}, min cost / '+\
                  f'real={np.round(self.objfun_min, 4):0.5f} / '+\
                  f'logged={np.round(best_val[1], 4):0.5f}{"":<3}@ {best_val[0]*self.log_interval: 5d}')
