# Support functions for TS project
# Author: Jacob Cybulski, ironfrown[at]gmail.com
# Aims: Provide support for quantum time series analysis
# Date: 2024

### Initial settings

import sys
sys.path.append('.')
sys.path.append('..')
sys.path

import matplotlib.pyplot as plt
import numpy as np
import pylab
import math
from IPython.display import clear_output

from matplotlib import set_loglevel
set_loglevel("error")

### Libraries used in QAE development

from qiskit import ClassicalRegister, QuantumRegister
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, TwoLocal, ZFeatureMap, ZZFeatureMap
from qiskit.circuit import Parameter


### Creates a circuit encoding a sequence 
#   - Adds H gates to position each qubit at a "zero" position
#   - Each sequence value y-rotates the qubit state either up (negative) or down (positive)
#   - Use only as many input values as the number of qubits, if not enough provided zero rotations are added
#
#   qubits_no: Number of qubits
#   input_no: Number of values in a sequence
#   optional classreg=False: Flag indicating if a classical register is to be added
#   optional meas_q=None: Indicates the qubit to be measured, active only when classreg=True
#   err_added: Indicates percent of error to be generated
#   returns: Circuit encoding a sequence
def sequence_encoder(qubit_no, input_no=None, classreg=False, meas_q=None, label='S'):

    if input_no == None: 
        input_no = qubit_no
    used_sgates = 0

    qr = QuantumRegister(qubit_no, 'q')
    cr = ClassicalRegister(1, 'meas')
    if classreg:
        seq = QuantumCircuit(qr, cr, name='sequence')
    else:
        seq = QuantumCircuit(qr, name='sequence')

    # Data-encoding circuit block, packs different input vars
    for q in range(qubit_no):
        seq.h(q)
        if q > input_no:
            seq.ry(0, q)
        else:
            param_x = Parameter(label+'('+str(used_sgates)+')') if used_sgates < input_no else 0
            seq.ry(param_x, q)
        used_sgates += 1

    if classreg and meas_q != None:
        seq.measure(meas_q, 0)

    return seq, seq.parameters[:]


### Creates an ansatz to be used for QAE Encoder/Decoder
#   num_latent: size of the latent area
#   num_trash: size of the trash area
#   reps: number of repeating layers
#   ent: type of entanglement layer (linear, reverse_linear, full, circular, sca, pairwise)

# Standard ansatz based Ry and Cx
def ansatz(num_qubits, reps=3, ent='sca', insert_barriers=False, label='A'):
    anz = RealAmplitudes(num_qubits, reps=reps, entanglement=ent, insert_barriers=insert_barriers,
                        parameter_prefix=label)
    return anz


### Swap test circuit
#   For two qubts, it returns a squared inner product between their states. 
#   In this implementation, it estimates the overlap between the states of all participating qubits. 
#   The repeated measurement of 1 indicates that the quantum states are identical. 
#   For two qubits, if the measurement returns 0.5 then the two states are orthogonal. 
#   However, orthogonality of the remaining states is no longer possible, so the measurements will be further away from 1, possibly confusing the outcome.
def swap_test(num_trash):
    qr = QuantumRegister(2 * num_trash + 1, "swap")
    cr = ClassicalRegister(1, "c")
    swap_qc = QuantumCircuit(qr, cr)
    auxiliary_qubit = 2 * num_trash
    
    swap_qc.h(auxiliary_qubit)
    for i in range(num_trash):
        swap_qc.cswap(auxiliary_qubit, i, num_trash + i)    
    swap_qc.h(auxiliary_qubit)
    swap_qc.measure(auxiliary_qubit, cr[0])
    return swap_qc

### Half-QAE Encoder/Decoder: Pure Input + Encoder/Decoder + Swap Test
#   Creates a encoder/decoder with input to be trained with the use of swap space on trash qubits.
#   To create a decoder pass inverse=True, later in testing to be reversed again.
def half_qae_encoder_with_swap_test(num_latent, num_trash, reps=4, ent='circular', inverse=False,
                  seq_name='Input', seq_label='N', 
                  anz_name='Encoder', anz_label='X'):
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    fm_qc, _ = sequence_encoder(num_latent+num_trash, label=seq_label)
    fm_qc.name = seq_name
    anz_qc = ansatz(num_latent+num_trash, reps=reps, ent=ent, label=anz_label).decompose()
    if inverse:
        anz_qc = anz_qc.inverse()
    anz_qc.name = anz_name
    swap_qc = swap_test(num_trash)
    swap_qc.name = 'Swap'
    
    qc = QuantumCircuit(qr, cr)
    qc.append(fm_qc, qargs=range(num_latent+num_trash))
    qc.barrier()
    qc.append(anz_qc, qargs=range(num_latent+num_trash))
    qc.barrier()
    qc.append(swap_qc, qargs=range(num_latent, num_latent + 2*num_trash+1), cargs=[0])
    qc.barrier()

    return qc, fm_qc.parameters, anz_qc.parameters


### Half-QAE Encoder with Sidekick Decoder$^\dagger$: Noisy Input + Encoder + Swap Test + Sidekick (Decoder$^\dagger$ + Pure Input)
#   Creates an encoder using the previously trained sidekick decoder$^\dagger$ by converging their common latent space.
#   The encoder is to be trained with noisy input and ensuring its latent space converges with that produced by the decoder$^\dagger$ from pure data.
def half_qae_encoder_with_sidekick(num_latent, num_trash, reps=4, ent='circular',
                  pure_seq_name='Pure Input', pure_seq_label='I', 
                  noisy_seq_name='Noisy Input', noisy_seq_label='N', 
                  enc_name='Encoder', enc_label='X',
                  dec_name='Decoder', dec_label='Y'):
    num_qubits = num_latent + num_trash
    qr = QuantumRegister(2 * num_qubits + 1, "q")
    cr = ClassicalRegister(1, "c")
    pure_in_qc, _ = sequence_encoder(num_qubits, label=pure_seq_label)
    pure_in_qc.name = pure_seq_name
    encoder_qc = ansatz(num_qubits, reps=reps, ent=ent, label=enc_label).decompose()
    encoder_qc.name = enc_name
    
    noisy_in_qc, _ = sequence_encoder(num_qubits, label=noisy_seq_label)
    noisy_in_qc.name = noisy_seq_name
    decoder_qc = ansatz(num_qubits, reps=reps, ent=ent, label=dec_label).decompose()
    decoder_qc = decoder_qc.inverse()
    decoder_qc.name = dec_name
    swap_qc = swap_test(num_latent)
    swap_qc.name = 'Swap'
    swap_qlist = list(range(num_latent))+list(range(num_qubits, num_qubits+num_latent))+[2*num_qubits]
    
    qc = QuantumCircuit(qr, cr)
    qc.append(pure_in_qc, qargs=range(num_qubits))
    qc.append(noisy_in_qc, qargs=range(num_qubits, 2*num_qubits))
    qc.barrier()
    qc.append(decoder_qc, qargs=range(num_qubits))
    qc.append(encoder_qc, qargs=range(num_qubits, 2*num_qubits))
    qc.barrier()
    qc.append(swap_qc, qargs=swap_qlist, cargs=[0])

    return qc, pure_in_qc.parameters, decoder_qc.parameters, noisy_in_qc.parameters, encoder_qc.parameters


### Full-QAE Stacked Encoder and previously trained Decoder$^\dagger$: Noisy Input + Encoder + Latent Space + Decoder + Pure Input$^\dagger$
#   Creates a full QAE to train its encoder using the previously trained decoder$^\dagger$ by converging the output to $\vert 0 \rangle^n$.
#   The encoder is to be trained with noisy input trough the latent space, then the decoder (no $^\dagger$), pure data$^\dagger$, to result in $\vert 0 \rangle^n$.
#   For training, all qubits of this Full-QAE will be measured.
def half_qae_encoder_stacked(num_latent, num_trash, reps=4, ent='circular',
                  pure_seq_name='Pure Input', pure_seq_label='I', 
                  noisy_seq_name='Noisy Input', noisy_seq_label='N', 
                  enc_name='Encoder', enc_label='X',
                  dec_name='Decoder', dec_label='Y'):
    num_qubits = num_latent + num_trash
    qr = QuantumRegister(num_qubits, "q")
    cr = ClassicalRegister(num_qubits, "c")
    pure_in_qc, _ = sequence_encoder(num_qubits, label=pure_seq_label)
    pure_in_qc.name = pure_seq_name
    pure_in_qc = pure_in_qc.inverse()
    encoder_qc = ansatz(num_qubits, reps=reps, ent=ent, label=enc_label).decompose()
    encoder_qc.name = enc_name
    
    noisy_in_qc, _ = sequence_encoder(num_qubits, label=noisy_seq_label)
    noisy_in_qc.name = noisy_seq_name
    decoder_qc = ansatz(num_qubits, reps=reps, ent=ent, label=dec_label).decompose()
    decoder_qc.name = dec_name
    
    qc = QuantumCircuit(qr, cr)
    qc.append(noisy_in_qc, qargs=range(num_qubits))
    qc.barrier()
    qc.append(encoder_qc, qargs=range(num_qubits))
    qc.barrier()
    for i in range(num_trash):
        qc.reset(num_latent + i)
    qc.barrier()
    qc.append(decoder_qc, qargs=range(num_qubits))
    qc.barrier()
    qc.append(pure_in_qc, qargs=range(num_qubits))
    qc.barrier()
    for i in range(num_qubits):
        qc.measure(qr[i], cr[i])

    return qc, pure_in_qc.parameters, decoder_qc.parameters, noisy_in_qc.parameters, encoder_qc.parameters


### Half-QAE Encoder: Input + Encoder without a Swap Test
#   Assesses the state of all trash qubits to be $\lvert 0 \rangle$ by their direct measurement and 
#   estimating the probability $P(\lvert 0 \rangle)^n$ (where $n$ is the number of qubits in the trash space. 
#   This approach to measuring qubit similarity is not as nuanced as what's is provided by swap test, 
#   as we miss on the state proximity determined by their inner product. 
#   However, the measurement is fast(er) and does not need additional qubits. 
#   The circuit training needs the cost function $cost = P(1-\lvert 0 \rangle)^n$, which needs to be minimised.
def half_qae_encoder(num_latent, num_trash, reps=4, ent='circular',
                  seq_name='Input', seq_label='N', 
                  anz_name='Encoder', anz_label='X'):
    qr = QuantumRegister(num_latent + num_trash, "q")
    cr = ClassicalRegister(num_trash, "c")
    fm_qc, _ = sequence_encoder(num_latent+num_trash, label=seq_label)
    fm_qc.name = seq_name
    anz_qc = ansatz(num_latent+num_trash, reps=reps, ent=ent, label=anz_label).decompose()
    anz_qc.name = anz_name

    
    circuit = QuantumCircuit(qr, cr)

    ### Sequence
    circuit.append(fm_qc, qargs=range(num_latent+num_trash))

    ### Encoder
    circuit.barrier()
    circuit.append(anz_qc, qargs=range(num_latent + num_trash))
    
    ### Measurements
    circuit.barrier()
    for i in range(num_trash):
        circuit.measure(qr[num_latent+i], cr[i])
    
    return circuit, fm_qc, anz_qc


### (Obsolete) Full-Circuit: Input + Encoder + Decoder + Output + No Swap Test
#   Note: Position of the output block may vary
#   Note: Decoder could be an inverse of encoder or become an independent block
def train_qae(num_latent, num_trash, reps=4, ent='sca',
              in_seq_name='Input', in_seq_label='I', 
              out_seq_name='Output', out_seq_label='O', 
              enc_name='Encoder', enc_label='X',
              dec_name='Decoder', dec_label='Y',
              keep_encoder=False):

    ### Create a circuit and its components
    lt_qubits = num_latent+num_trash
    qr = QuantumRegister(lt_qubits, "q")
    cr = ClassicalRegister(lt_qubits, "c")
    in_qc, _ = sequence_encoder(lt_qubits, label=in_seq_label)
    in_qc.name = in_seq_name
    out_qc, _ = sequence_encoder(lt_qubits, label=out_seq_label)
    out_qc.name = out_seq_name
    enc_qc = ansatz(lt_qubits, reps=reps, ent=ent, label=enc_label).decompose()
    enc_qc.name = enc_name
    dec_qc = ansatz(lt_qubits, reps=reps, ent=ent, label=dec_label).decompose()
    dec_qc.name = dec_name

    ### Input
    qc = QuantumCircuit(qr, cr)
    qc.append(in_qc, qargs=range(lt_qubits))

    ### Encoder
    qc.barrier()
    qc.append(enc_qc, qargs=range(lt_qubits))

    ### Latent / Trash
    qc.barrier()
    for i in range(num_trash):
        qc.reset(num_latent + i)

    ### Decoder
    qc.barrier()
    if keep_encoder:
        dec_inv_qc = enc_qc.inverse()
    else:
        dec_inv_qc = dec_qc.inverse()
    qc.append(dec_inv_qc, qargs=range(lt_qubits))

    ### Inverted output (trans input)
    qc.barrier()
    out_inv_qc = out_qc.inverse()
    qc.append(out_inv_qc, qargs=range(lt_qubits))

    ### Measurements
    qc.barrier()
    for i in range(len(qc.qubits)):
        qc.measure(qr[i], cr[i])
    
    ### Collect weight parameters
    train_weight_params = []
    for enc_p in enc_qc.parameters:
        train_weight_params.append(enc_p)    
    if not keep_encoder:
        for dec_p in dec_qc.parameters:
            train_weight_params.append(dec_p)

    ### Collect in/out parameters
    in_out_params = []
    for in_p in in_qc.parameters:
        in_out_params.append(in_p)    
    for out_p in out_qc.parameters:
        in_out_params.append(out_p)        
    
    if keep_encoder:
        return qc, in_out_params, enc_qc.parameters, enc_qc.parameters, train_weight_params
    else:
        return qc, in_out_params, enc_qc.parameters, dec_qc.parameters, train_weight_params


### Testing Circuit - The entire QAE
#   The full Autoencoder consists of both the Encoder and Decoder, which is simply an inverted Encoder. 
#   Both the Encoder and Decoder can be initialised using the same parameters obtained from the Encoder (plus swap test) training.
#   By applying the full QAE circuit to a test dataset, we can then determine the model accuracy.
def qae(lat_no, trash_no, reps=2, ent='sca', 
        classreg=False, meas_q=None, keep_encoder=False, invert_dec=True,
        in_seq_name='Noisy In', in_seq_label='N', 
        enc_name='Encoder', enc_label='X',
        dec_name='Decoder', dec_label='Y'):

    # Prepare a circuit
    qr = QuantumRegister(lat_no + trash_no, 'q')
    cr = ClassicalRegister(1, 'meas')
    # qae_qc = QuantumCircuit(lat_no + trash_no, 1)
    if classreg:
        qae_qc = QuantumCircuit(qr, cr, name='qae')
    else:
        qae_qc = QuantumCircuit(qr, name='qae')

    # Create all QAE components
    in_qc, _ = sequence_encoder(lat_no + trash_no, label=in_seq_label)
    in_qc.name = in_seq_name
    enc_qc = ansatz(lat_no+trash_no, reps=reps, ent=ent, label=enc_label)
    enc_qc.name = enc_name
    dec_qc = ansatz(lat_no+trash_no, reps=reps, ent=ent, label=dec_label)
    dec_qc.name = dec_name

    # Create a circuit
    qae_qc.append(in_qc, qargs=range(lat_no + trash_no))
    qae_qc.barrier()
    qae_qc.append(enc_qc, qargs=range(lat_no + trash_no))
    qae_qc.barrier()
    
    for i in range(trash_no):
        qae_qc.reset(lat_no + i)
    
    qae_qc.barrier()
    if keep_encoder:
        dec_inv_qc = enc_qc.inverse()
    elif invert_dec:
        dec_inv_qc = dec_qc.inverse()
    else:
        dec_inv_qc = dec_qc
    qae_qc.append(dec_inv_qc, qargs=range(lat_no + trash_no))

    # Add optional measurement
    if classreg and meas_q != None:
        qae_qc.barrier()
        qae_qc.measure(meas_q, 0)

    return qae_qc, in_qc, enc_qc, dec_qc
