# Support functions for Qiskit models creation
# Author: Jacob Cybulski, ironfrown[at]gmail.com
# Aims: Provide support for QTSA model creation
# Date: 2024-2025

import os
import numpy as np
import math

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit_algorithms.optimizers import L_BFGS_B, P_BFGS, COBYLA, NELDER_MEAD, QNSPSA, ADAM, UMDA
from qiskit.circuit.library import RealAmplitudes, TwoLocal, ZFeatureMap, ZZFeatureMap,EfficientSU2, PauliFeatureMap
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_algorithms.utils import algorithm_globals
from qiskit.visualization import plot_histogram, plot_state_city, plot_state_paulivec
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor, VQR
from qiskit_machine_learning.utils.loss_functions import L1Loss, L2Loss
from qiskit.circuit import Parameter
from qiskit.compiler import transpile


### Create a custom serial model circuit 
#   - To be used with CircuitQNN / NeuralNetworkRegressor

def serial_fourier_model(layers, add_meas=False):
    
    ansatz = QuantumCircuit(1, name="ansatz")
    param_x = Parameter('X')
    pno = 0 # Variational parameter counter
    params = []

    # Data encoding block
    def S():
        ansatz.rx(param_x, 0)

    # Trainable variational block
    def W(layer, label):
        nonlocal pno
        nonlocal params
        param_w_0 = Parameter(f'{label}[{pno:03d}]')
        param_w_1 = Parameter(f'{label}[{pno+1:03d}]')
        param_w_2 = Parameter(f'{label}[{pno+2:03d}]')
        params += [param_w_0, param_w_1, param_w_2]
        ansatz.u(param_w_0, param_w_1, param_w_2, 0)
        pno += 3

    # Create layers of W, S blocks
    for l in range(layers):
        W(l, 'W')
        S()

    # Add the final block
    W(layers, 'W')

    if add_meas:
        ansatz.measure_all()

    # Create a parameter list
    params += [param_x]

    return ansatz 


### Create a custom parallel model circuit 
#   - To be used with CircuitQNN / NeuralNetworkRegressor

def parallel_fourier_model(qubit_no, before_layers, after_layers, add_meas=False):
    
    qr = QuantumRegister(qubit_no, 'q')
    ansatz = QuantumCircuit(qr, name='ansatz')
    param_x = Parameter('x')

    # Data encoding block
    def S():
        for q in range(qubit_no):
            ansatz.rx(param_x, q)

    # Trainable variational block
    def W(layers, label):
        pno = 0  # Variational parameter counter
        
        for l in range(layers):
            ansatz.barrier()
            for q in range(qubit_no):
                ansatz.u(Parameter(f'{label}[{pno:03d}]'), 
                         Parameter(f'{label}[{pno+1:03d}]'), 
                         Parameter(f'{label}[{pno+2:03d}]'), 
                         q)
                pno += 3
            ansatz.barrier()
            ansatz.append(
                TwoLocal(qubit_no, [], 'cx', 
                         entanglement='circular',
                         reps=1, 
                         parameter_prefix=label, 
                         insert_barriers=False,
                         skip_final_rotation_layer=False),
                qargs=qr)
        
        ansatz.barrier()
        for q in range(qubit_no):
            ansatz.u(Parameter(f'{label}[{pno:03d}]'), 
                     Parameter(f'{label}[{pno+1:03d}]'), 
                     Parameter(f'{label}[{pno+2:03d}]'), 
                     q)
            pno += 3

    
    # Create layers of WB, S, WA blocks
    W(before_layers, 'B')
    ansatz.barrier()
    S()
    W(after_layers, 'A')

    if add_meas:
        ansatz.measure_all()

    return ansatz.decompose().decompose()    


def parallel_fourier_model_with_reuploading(qubit_no, before_layers, after_layers, add_meas=False):
    
    qr = QuantumRegister(qubit_no, 'q')
    ansatz = QuantumCircuit(qr, name='ansatz')
    param_x = Parameter('x')

    # Data encoding block
    def S():
        for q in range(qubit_no):
            ansatz.rx(param_x, q)

    # Data encoding block
    def BS(layers, label):
        pno = 0  # Variational parameter counter
        
        for l in range(layers):
            ansatz.barrier()
            for q in range(qubit_no):
                ansatz.u(Parameter(f'{label}[{pno:03d}]'), 
                         Parameter(f'{label}[{pno+1:03d}]'), 
                         Parameter(f'{label}[{pno+2:03d}]'), 
                         q)
                pno += 3
            ansatz.barrier()
            if qubit_no > 1:
                ansatz.append(
                    TwoLocal(qubit_no, [], 'cx', 
                             entanglement='circular',
                             reps=1, 
                             parameter_prefix=label, 
                             insert_barriers=False,
                             skip_final_rotation_layer=False),
                    qargs=qr)     
                ansatz.barrier()
            S()
            
    # Trainable variational block
    def AW(layers, label):
        pno = 0  # Variational parameter counter
        
        for l in range(layers):
            ansatz.barrier()
            for q in range(qubit_no):
                ansatz.u(Parameter(f'{label}[{pno:03d}]'), 
                         Parameter(f'{label}[{pno+1:03d}]'), 
                         Parameter(f'{label}[{pno+2:03d}]'), 
                         q)
                pno += 3
            if qubit_no > 1:
                ansatz.barrier()
                ansatz.append(
                    TwoLocal(qubit_no, [], 'cx', 
                             entanglement='circular',
                             reps=1, 
                             parameter_prefix=label, 
                             insert_barriers=False,
                             skip_final_rotation_layer=False),
                    qargs=qr)
        
        ansatz.barrier()
        for q in range(qubit_no):
            ansatz.u(Parameter(f'{label}[{pno:03d}]'), 
                     Parameter(f'{label}[{pno+1:03d}]'), 
                     Parameter(f'{label}[{pno+2:03d}]'), 
                     q)
            pno += 3

    
    # Create layers of WB, S, WA blocks
    BS(before_layers, 'B')
    AW(after_layers, 'A')

    if add_meas:
        ansatz.measure_all()

    return ansatz.decompose().decompose()    


### Serial sliding window ansatz
#   - To be used with CircuitQNN / NeuralNetworkRegressor
#
#   inputs
#     qubit_no: The number of qubits in the circuit
#     input_no: The number of input values / the size of a sliding window
#     xlayers: The number of "external" ansatz layers
#     add_meas: If true measurements will be added
#     ent: Types of entanglements to be used
#
#   returns
#     Circuit: The serial sliding window ansatz 
#     SParams: The list of parameters used in all S blocks
#     WParams: The list of parameters used in all W blocks
#     Layers: The calculated number of "internal" layers in the circuit

def swindow_serial_model(qubit_no, input_no, xlayers=1, add_meas=False, ent='circular'):

    SLabel = 'X'
    WLabel = 'W'
    
    ilayers = input_no // qubit_no + (0 if input_no % qubit_no == 0 else 1)
    used_sgates = 0

    # Predefined all S parameters
    SParams = []
    for q in range(input_no):
        param_x = Parameter(f'{SLabel}[{q:03d}]')
        SParams.append(param_x)

    ### Create an ansatz
    qr = QuantumRegister(qubit_no, 'q')
    ansatz = QuantumCircuit(qr, name='ansatz')
    WParams = []

    # Data-encoding circuit block, packs different input vars
    def SN(qubit_no):
        nonlocal used_sgates, SParams
        s_qr = QuantumRegister(qubit_no)
        s_ansatz = QuantumCircuit(s_qr, name='SSM')
        for q in range(qubit_no):
            param_x = SParams[q] if used_sgates < input_no else 0
            s_ansatz.rx(param_x, q)
            used_sgates += 1
        return s_ansatz

    # Trainable circuit block
    def W(qubit_no, label):
        w_qr = QuantumRegister(qubit_no)
        w_ansatz = QuantumCircuit(w_qr, name='W')
        w_ansatz.append(
            TwoLocal(qubit_no, ['rx', 'ry', 'rz'], 'cx', 
                     entanglement=ent,
                     reps=1, 
                     parameter_prefix=label, 
                     insert_barriers=True,
                     skip_final_rotation_layer=False),
            qargs=w_qr)
        return w_ansatz

    used_wlayers = 0
    for xl in range(xlayers):
        used_sgates = 0
        
        for il in range(ilayers):
            W1 = W(qubit_no, f'W_{used_wlayers:02d}')
            used_wlayers += 1
            WParams = WParams + W1.parameters[:]
            ansatz.append(W1, qargs=qr)
            ansatz.barrier()
            SX = SN(qubit_no) # , 'X'+str(il))
            SParams = SParams + SX.parameters[:]
            ansatz.append(SX, qargs=qr)
            ansatz.barrier()
    
        if xl == xlayers-1:
            W1 = W(qubit_no, f'W_{used_wlayers:02d}')  
            used_wlayers += 1
            WParams = WParams + W1.parameters[:]
            ansatz.append(W1, qargs=qr)

    if add_meas:
        ansatz.measure_all()

    return ansatz.decompose().decompose().decompose(), SParams, WParams, ilayers


### Sliding window QNN model
# observable = global / local / patrial
def swindow_qnn_model(qubits_no, inputs_no, fm_layers_no, ans_layers_no, ent='full', 
                      insert_barriers=True, add_meas=False, return_components=False,
                      observable='global'):

    fm_map = ZZFeatureMap(inputs_no, reps=fm_layers_no, insert_barriers=insert_barriers, parameter_prefix='s')
    # ansatz = RealAmplitudes(qubits_no, entanglement=ent, reps=ans_layers_no, insert_barriers=insert_barriers, parameter_prefix='w')
    ansatz = TwoLocal(qubits_no, ['rx', 'ry', 'rz'], 'cx', 
             entanglement=ent,
             reps=ans_layers_no, 
             parameter_prefix='w', 
             insert_barriers=True,
             skip_final_rotation_layer=False)
    fm_start_qubit = (qubits_no-inputs_no)//2
    fm_end_qubit = fm_start_qubit+inputs_no
    
    if add_meas:
        if observable == 'local':
            circ = QuantumCircuit(qubits_no, 1, name="circ")
        elif observable == 'global':
            circ = QuantumCircuit(qubits_no, qubits_no, name="circ")
        else:
            circ = QuantumCircuit(qubits_no, inputs_no, name="circ")
    else:
        circ = QuantumCircuit(qubits_no, name="circ")

    if qubits_no < inputs_no:
        return circ, fm_map.parameters, ansatz.parameters

    circ.append(fm_map, qargs=range(fm_start_qubit, fm_end_qubit))
    circ.barrier()
    circ.append(ansatz, qargs=ansatz.qubits)
    if add_meas:
        if observable == 'local':
            circ.measure(fm_start_qubit+inputs_no//2, 0)
        elif observable == 'global':
            # This is for observable Z*qubits_no 
            for q in range(qubits_no):
                circ.measure(q, q)
        else: # observable == 'input'
            # This is for semi-localised observable (Z*inputs_no, I*(qubits_no-inputs_no)
            for q in range(inputs_no):
                circ.measure(fm_start_qubit+q, q)
    if return_components:
        return fm_map, ansatz, circ
    else:
        return circ.decompose().decompose(), fm_map.parameters, ansatz.parameters


import torch
from torch import nn, tensor, optim

### Classic large estimator
class Classic_NN(nn.Module):

    def __init__(self, in_shape, out_shape):
        super(Classic_NN, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(in_shape, 150),
            nn.LeakyReLU(True),
            #nn.Dropout(0.2),
            nn.Linear(150, 100),
            nn.LeakyReLU(True),
            #nn.Dropout(0.2),
            nn.Linear(100, 50),
            #nn.BatchNorm1d(50),
            nn.LeakyReLU(True),
            #nn.Dropout(0.2),
            nn.Linear(50, out_shape)
        )
        
    def forward(self, x):
        x = self.model(x)
        return x

### Classic medium estimator
class Classic_NN_Medium(nn.Module):

    def __init__(self, in_shape, out_shape):
        super(Classic_NN_Medium, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(in_shape, 50),
            nn.LeakyReLU(True),
            nn.Linear(50, 80),
            nn.BatchNorm1d(80),
            nn.LeakyReLU(True),
            nn.Linear(80, 20),
            nn.LeakyReLU(True),
            nn.Linear(20, out_shape)
        )
        
    def forward(self, x):
        x = self.model(x)
        return x

### Classic small estimator
class Classic_NN_Small(nn.Module):

    def __init__(self, in_shape, out_shape):
        super(Classic_NN_Small, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(in_shape, 10),
            nn.ReLU(True),
            # nn.Dropout(0.01),
            nn.Linear(10, 15),
            nn.BatchNorm1d(15),
            nn.ReLU(True),
            # nn.Dropout(0.01),
            nn.Linear(15, 10),
            nn.ReLU(True),
            # nn.Dropout(0.01),
            nn.Linear(10, out_shape)
        )
        
    def forward(self, x):
        x = self.model(x)
        return x

### Classic tiny estimator
class Classic_NN_Tiny(nn.Module):

    def __init__(self, in_shape, out_shape):
        super(Classic_NN_Tiny, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(in_shape, 8),
            nn.ReLU(True),
            # nn.Dropout(0.01),
            nn.Linear(8, 5),
            nn.BatchNorm1d(5),
            nn.ReLU(True),
            # nn.Dropout(0.01),
            nn.Linear(5, 3),
            nn.BatchNorm1d(3),
            nn.ReLU(True),
            # nn.Dropout(0.01),
            nn.Linear(3, out_shape)
        )
        
    def forward(self, x):
        x = self.model(x)
        return x