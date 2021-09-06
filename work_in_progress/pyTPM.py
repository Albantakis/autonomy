
import numpy as np
import os
import numpy.random as ran
import copy
import pyphi
from pathlib import Path
import scipy.io as sio
import pandas as pd

### Todo: CHeck that TPM is actually correct and that nodes are ordered Input-Motor-Hidden in MABE.

def get_genome(genomes, run, agent):
    genome = genomes[run]['GENOME_root::_sites'][agent]
    genome = np.squeeze(np.array(np.matrix(genome)))
    return genome

def genome2TPM(genome, n_nodes=8, n_sensors=2, n_motors=2, gate_type='deterministic',states_convention='loli',remove_sensor_motor_effects=True):
    '''
    Extracts the TPM from the genome output by mabe.
        Inputs:
            genome: np.array of the genome output from mabe (1 x n_codons)
            n_nodes:(maximal) number of nodes in the agent
            gate_type: string specifying the type of gates coded by the genome ('deterministic' or 'decomposable')
            states_convention: string specifying the convention used for ordering of the states ('holi' or 'loli')

        Outputs:
            TPM: The full state transition matrix for the agent (states x nodes)
            gate_TPMs: the TPM for each gate specified by the genome (which in turn specifies the full TPM)
            cm: Connectivity matrix (nodes x nodes) for the agent. 1's indicate there is a connection, 0's indicate the opposite
    '''

    # Setting max number of inputs and outputs (defined in MABE settings)
    max_inputs = max_outputs = max_io = 4 # 4 inputs, 4 outputs per HMG

    # Defining start codon and gene length depending on gate type used
    if gate_type=='deterministic':
        # max_gene_length = 300
        max_gene_length = 12 + (2**max_inputs)*max_outputs + 1 # = 77
        start_codon = 43
        print(start_codon)

    elif gate_type=='decomposable':
        # max_gene_length = 400
        max_gene_length = 12 + (2**max_inputs)*max_outputs + 1 #+ (max_io**4)
        start_codon = 52

    else:
        raise AttributeError("Unknown gate type.")

    ixs = np.where(genome==start_codon)[0]

    # making sure start codon is not the last codon in the gene
    ixs = [ix for ix in ixs if ix+1<len(genome)]

    # checking if next codon matches the "second start codon" required
    gene_ixs = [ix for ix in ixs if genome[ix+1]==255-start_codon]

    # although genome is circular, it is treated as if it is linear (no gene can cross from the last to first codon)
    gene_ixs = [ix for ix in gene_ixs if ix + max_gene_length <= len(genome)] # genome is finite (even though genomeType=Circular)

    # Storing all reading frames (genes) following a qualified start codon
    genes = np.array([genome[ix:ix+max_gene_length] for ix in gene_ixs])
    n_genes = genes.shape[0]

    # -----------------------
    # read out genes
    # -----------------------
    # locations:
    # 2    number of inputs
    # 3    number of outputs
    # 4-7  possible inputs
    # 8-11 possible outputs
    # 12   TPM

    # initializing output structures
    cm = np.zeros((n_nodes,n_nodes))
    full_TPM = np.zeros((2**n_nodes, n_nodes, n_genes))
    gate_TPMs = []

    # looping over, and parsing Markov gates from genome
    for i, gene in zip(range(n_genes),genes):
        # print('Gene: {}/{}'.format(i+1,n_genes))

        # Get gate's inputs and outputs
        n_inputs = gene[2] % max_inputs + 1
        n_outputs = gene[3] % max_outputs + 1
        raw_inputs = gene[4:4+n_inputs]
        raw_outputs = gene[8:8+n_outputs]
        inputs = gene[4:4+n_inputs] % n_nodes
        outputs = gene[8:8+n_outputs] % n_nodes

        # Get probabilities
        gate_TPM = np.zeros((2**n_inputs,n_outputs))
        for row in range(2**n_inputs):
            if start_codon == 52: # Decomposable
                # start_locus = 12 + row * n_outputs + (max_io**4)
                start_locus = 12 + row * max_outputs
                raw_probability = gene[start_locus:start_locus+n_outputs]
                gate_TPM[row,:] = raw_probability/255. # or 256?

            elif start_codon == 43: # (Deterministic)
                start_locus = 12 + row * max_outputs
                raw_probability = gene[start_locus:start_locus+n_outputs]
                gate_TPM[row,:] = raw_probability % 2

        # Reduce gate's degenerate outputs (if there are)
        if start_codon == 52: # Decomposable
            g_TPM, outputs = reduce_degenerate_outputs(1-gate_TPM, outputs)
            g_TPM, inputs  = reduce_degenerate_inputs(g_TPM, inputs, states_convention)
            gate_TPMs.append({'type': gate_type,
                              'ins': inputs,
                              'outs': outputs,
                              'logic': (1-g_TPM).tolist()})
        elif start_codon == 43: # (Deterministic)
            gate_TPM, outputs = reduce_degenerate_outputs(gate_TPM, outputs)
            gate_TPM, inputs  = reduce_degenerate_inputs(gate_TPM, inputs, states_convention)
            gate_TPMs.append({'type': gate_type,
                              'ins': inputs,
                              'outs': outputs,
                              'logic': gate_TPM.tolist()})

        # Get list of all possible states from nodes
        cm[np.ix_(inputs,outputs)] = 1

        # Expand gate TPM to the size of the full state-by-node TPM for combination
        if start_codon == 52: # Decomposable
            full_TPM[:,:,i] = expand_gate_TPM(g_TPM, inputs, outputs, n_nodes, states_convention)
        elif start_codon == 43: # (Deterministic)
            full_TPM[:,:,i] = expand_gate_TPM(gate_TPM, inputs, outputs, n_nodes, states_convention)

    # combining the gate TPMs to the full TPM (assuming independence)
    TPM = 1 - np.prod(1 - full_TPM,2)

    # Removing the effects of feedback to sensors and from motores.
    # These connections are ineffective in MABE due to the way  Complexiphi world is implemented
    if remove_sensor_motor_effects:
        TPM = remove_motor_sensor_effects(TPM,n_sensors,n_motors,n_nodes)
        cm = remove_motor_sensor_connections(cm,n_sensors,n_motors)

    # Return outputs
    return TPM, gate_TPMs, cm

def genome2TPM_combined(genome,n_nodes=8, n_sensors=2, n_motors=2, gate_types=['deterministic','decomposable']):
    '''
    Function for parsing genomes containing multiple gate types.
        Inputs:
            n_nodes: number of nodes to calculate the full state state matrix for
            convention: 'loli' (little-endian) or 'holi' (big-endian)) state labelling convention
        Outputs:
            states: state by node (2**n x n) array containing all binary states
    '''

    full_TPM = np.zeros((2**n_nodes, n_nodes, len(gate_types)))
    full_CM = np.zeros((n_nodes, n_nodes, len(gate_types)))
    if type(gate_types)==list:
        for i in range(len(gate_types)):
            full_TPM[:,:,i], full_gates, full_CM[:,:,i] = genome2TPM(genome, n_nodes=8, n_sensors=2, n_motors=2, gate_type=gate_types[i])

        full_TPM = 1 - np.prod(1 - full_TPM, 2)
        full_TPM = remove_motor_sensor_effects(full_TPM,n_sensors,n_motors,n_nodes)
        full_CM = np.sum(full_CM,2)
    elif type(gate_types)==str:
        full_TPM, full_gates, full_CM = genome2TPM(genome, n_nodes=8, n_sensors=2, n_motors=2, gate_type=gate_types)
    else:
        print('strange gate type')

    return full_TPM, full_CM

def genome2TPM_from_csv(path, agent, n_nodes=8, n_sensors=2, n_motors=2, gate_type='deterministic',states_convention='loli',remove_sensor_motor_effects=True):
    '''
    Create a genome directly from a MABE csv output.
    '''
    genome_data = pd.read_csv(path)
    genome = np.squeeze(np.array(np.matrix(genome_data['GENOME_root::_sites'][agent])))

    return genome2TPM(genome, n_nodes, n_sensors, n_motors, gate_type,states_convention,remove_sensor_motor_effects)

def reduce_degenerate_outputs(gate_TPM, outputs):
    '''
    Reduces gate_TPM with degenerate outputs (e.g. outputs=[2,12,3,12] to outputs=[2,3,12]) by combining
    them with OR logic
        Inputs:
            gate_TPM: Original gate TPM (states x nodes) to be reduced
            outputs: IDs for the outputs the gate connects to (1 x nodes)
        Outputs:
            reduced_gate_TPM: Reduced gate TPM (states x nodes) now without degenerate outputs
            unique_outputs: IDs for the unique nodes the gate connects to (1 x nodes)
    '''
    # Find degenerate outputs
    unique_outputs = np.unique(outputs)
    unique_ixs = []

    # return immediately if there are no degenerate outputs
    if len(outputs)==len(unique_outputs):
        return gate_TPM, outputs

    # make list indicating the index of the first occurrence of each output
    for e in unique_outputs:
        ixs = list(np.where(outputs==e)[0])
        unique_ixs.append(ixs)

    # Reduce the effect on outputs using OR logic (outputs of gate are independent)
    reduced_gate_TPM = np.zeros((gate_TPM.shape[0],len(unique_outputs)))
    for i in range(len(unique_outputs)):
        reduced_gate_TPM[:,i] = 1 - np.prod(1 - gate_TPM[:,unique_ixs[i]],1) # OR logic

    # Return outputs
    return reduced_gate_TPM, unique_outputs

def reduce_degenerate_inputs(gate_TPM, inputs, states_convention):
    '''
    Function for reducing gate_TPM with degenerate inputs (e.g. inputs=[2,12,3,12] to inputs=[2,3,12]) by removing
    input states that are internally inconsistent.
        Inputs:
            gate_TPM: the original gateTPM (states x nodes)
            inputs: IDs of inputs to the gate (1 x nodes)
            states_convention: specification of the covention used for state organizatino (loli or holi)
        Outputs:
            reduced_gate_TPM: the reduced gateTPM (states x nodes), now without degenerate inputs
            unique_inputs: IDs of unique inputs to the gate (1 x nodes)
            '''
    # Find degenerate inputs
    inputs = np.array(inputs)
    unique_inputs = np.unique(inputs)
    unique_ixs = []

    # Returning immediatley if no degenerate inputs exist
    if len(unique_inputs)==len(inputs):
        return gate_TPM, inputs

    # make list indicating the index of the first occurrence of each input
    for e in unique_inputs:
        ixs = list(np.where(inputs==e)[0])
        unique_ixs.append(ixs)

    # creating the logic table (all binary states) using same convention as the state-by-node TPM
    input_states = get_states(len(inputs),convention=states_convention)

    # finding states where same node gives different values (inconsistent states)
    keepcols_ixs = [ixs[0] for ixs in unique_ixs]
    keepcols_ixs = np.sort(keepcols_ixs)
    delete_row = []

    for ixs in unique_ixs:
        # check for duplicates
        if len(ixs)>1:
            # run through all states
            for i in list(range(len(input_states))):
                state = input_states[i,ixs]
                # check if activity of all duplicates match
                if not ((np.sum(state) == len(state)) or (np.sum(state) == 0)):
                    # remove row when they do not match
                    delete_row.append(i)
    reduced_gate_TPM = np.delete(gate_TPM,(delete_row),axis=0)

    # Reorder rows after deletion of degenerate inputs
    input_states = np.delete(input_states, (delete_row),axis=0)
    input_states = input_states[:,keepcols_ixs]
    ixs = []

    # finding the correct row index for the remaining rows
    for state in input_states:
        ixs.append(pyphi.convert.s2l(state))
    ixs_order = np.argsort(ixs)

    # Setting gate TPM in the right order, only including consistent states
    final_gate_TPM = reduced_gate_TPM[ixs_order,:]
    final_inputs = inputs[keepcols_ixs]

    # Returning Outputs
    return final_gate_TPM, final_inputs


def expand_gate_TPM(gate_TPM, inputs, outputs, n_nodes, states_convention):
    '''
    Function for expanding the gate TPM (2**n_inputs x n_outputs) to the full size TPM (2**n_nodes x n_nodes).
        Inputs:
            gate_TPM: Original gate TPM to be expanded (input-states x output-nodes)
            inputs: IDs of inputs (1 x nodes)
            outputs: IDs of outputs (1 x nodes)
            n_nodes: total number of nodes in the agent
            states_convention: specification of convention used for state organization ('holi' or 'loli')
        Outputs:
            expanded_gate_TPM: Final gate TPM expanded to the size of the full agent TPM (system-states x system-nodes)
    '''
    # getting the states in the order defined by states_convention for both the full network and the nodes involved in the gate
    full_states = get_states(n_nodes, convention=states_convention)
    gate_states = get_states(len(inputs), convention=states_convention)

    # initializing the expanded TPM
    expanded_gate_TPM = np.zeros((2**n_nodes,n_nodes))

    # iterate through rows of the expanded gate TPM
    for i in range(expanded_gate_TPM.shape[0]):
        # finding the state of input nodes that matches the full state
        for j in range(gate_TPM.shape[0]):
            if np.all(gate_states[j,:]==full_states[i,inputs]):
                expanded_gate_TPM[i,outputs] = gate_TPM[j,:]
                break

    # Returning outputs
    return expanded_gate_TPM


def remove_motor_sensor_effects(TPM,n_sensors=2,n_motors=2,n_nodes=4,states_convention = 'loli'):
    '''
        Removes effects of hidden and motor neurons on sensors (sensors always transition to 0s in next state) and removing
        feedback of motor to hidden neurons (hidden neuron states are conditionally independent on motors states in t-1).
        Inputs:
            TPM: the state-by-node TPM of the system (np.array, nodes**2 x nodes),
            n_sensors: number of sensors in the system (int),
            n_motors: number of motors in the system (int),
            n_nodes: number of hidden nodes in the system (int),
            states_convention: specification of convention used for state organization (string, 'holi' or 'loli')
        Outputs:
            TPM: the updated state-by-node TPM of the system (np.array, nodes x nodes**2)
    '''
    # forcing all sensors to be zero in the effect
    TPM[:,0:n_sensors] = np.ones(np.shape(TPM[:,0:n_sensors]))/2.

    # converting TPM to multidimensional representation for easier calling
    TPM_multi = pyphi.convert.to_multidimensional(TPM)

    # setting all output states to be identical to the state with motors being off (forcing the effect of motors to be null)
    # first splitting the system states into motor states and non-motor states
    no_motor_states = get_states(n_nodes-n_motors, states_convention)
    motor_states = get_states(n_motors, states_convention)

    # looping through all non-motor states
    for state in no_motor_states:
        sensors = list(state[:n_sensors])
        hidden = list(state[n_sensors:])
        full_state = tuple(sensors+list(motor_states[0,:])+hidden) # all motors off state
        next_state = TPM_multi[full_state] # TP for the motors-off state

        # Looping through all motor states, and setting the relevant row in the full TPM to the TP found above
        for motor_state in motor_states[1:]:
            full_state = tuple(sensors+list(motor_state)+hidden)
            TPM_multi[full_state] = next_state

    # Converting TPM back to 2D format
    TPM = pyphi.convert.to_2dimensional(TPM_multi)

    # returning outputs
    return TPM


def remove_motor_sensor_connections(cm,n_sensors=2,n_motors=2):
    '''
    Function for removing the apparent connections to sensors and from motors
        Inputs:
            cm: connectivity matrix for a system (nodes x nodes)
            n_sensors: number of sensors in the system (int),
            n_motors: number of motors in the system (int),

        Outputs:
            cm: updated connectivity matrix (nodes x nodes)

    '''

    # setting all connections to sensors to 0
    cm[:,0:n_sensors] = np.zeros(np.shape(cm[:,0:n_sensors]))
    # setting all connections from motors to 0
    cm[n_sensors:n_sensors+n_motors] = np.zeros(np.shape(cm[n_sensors:n_sensors+n_motors]))

    # returning output
    return cm


def get_states(n_nodes, convention='loli'):
    '''
    Function for generating arrays with all possible states according to holi and loli indexing conventions.
        Inputs:
            n_nodes: number of nodes to calculate the full state state matrix for
            convention: 'loli' (little-endian) or 'holi' (big-endian)) state labelling convention
        Outputs:
            states: state by node (2**n x n) array containing all binary states
    '''
    # building the states matrix by generating a list of binary numbers from 0 to 2**nodes - 1
    states_holi = np.array(([list(('{:0'+str(n_nodes)+'d}').format(int(bin(x)[2:]))) for x in range(2**n_nodes)])).astype(int)
    states_loli = np.flip(states_holi,1)

    # Returning outputs
    if convention=='loli':
        return states_loli
    else:
        return states_holi

### THIS FUNCTION IS NOT TESTED ###
def gates2TPM(gates,n_nodes,states_convention='loli',remove_sensor_motor_effects=False):
    '''
    Builds genome given gate-TPMs.
        Inputs:

        Outputs:

    '''
    n_gates = len(gates)
    cm = np.zeros((n_nodes,n_nodes))
    full_TPM = np.zeros((2**n_nodes, n_nodes, n_gates))
    gate_TPMs = []
    for i,gate in zip(range(n_gates),gates):

        # Get gate's inputs and outputs
        inputs = gate['ins']
        outputs = gate['outs']

        # Get list of all possible states from nodes
        cm[np.ix_(inputs,outputs)] = 1

        # Reduce gate's degenerate outputs (if there are)
        gate_TPM = np.array(gate['logic'])
        gate_TPM, outputs = reduce_degenerate_outputs(gate_TPM, outputs)
        gate_TPM, inputs = reduce_degenerate_inputs(gate_TPM, inputs, states_convention)

        gate_TPMs.append({'type': gate['type'],
                          'ins': inputs,
                          'outs': outputs,
                          'logic': gate_TPM.tolist()})
        # Expand gate TPM
        full_TPM[:,:,i] = expand_gate_TPM(gate_TPM, inputs, outputs, n_nodes, states_convention)

    TPM = 1 - np.prod(1 - full_TPM,2)

    if remove_sensor_motor_effects:
        TPM = remove_motor_sensor_effects(TPM,n_sensors,n_motors,n_nodes)
        cm = remove_motor_sensor_connections(cm,n_sensors,n_motors)

    print('Done.')
