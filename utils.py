import numpy as np
import networkx as nx
from copy import copy
import pandas as pd

# General 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_node_labels(agent):
    node_labels = []

    # standard labels for up to 10 nodes of each kind
    sensor_labels = ['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10']
    motor_labels = ['M1','M2','M3','M4','M5','M6','M7','M8','M9','M10']
    hidden_labels = ['A','B','C','D','E','F','G','H','I','J']

    # defining labels for each node type
    s = [sensor_labels[i] for i in list(range(agent.n_sensors))]
    m = [motor_labels[i] for i in list(range(agent.n_motors))]
    h = [hidden_labels[i] for i in list(range(agent.n_hidden))]

   # combining the labels
    node_labels.extend(s)
    node_labels.extend(m)
    node_labels.extend(h)

    #sort node labels according to idx
    indices = agent.sensor_ixs + agent.motor_ixs + agent.hidden_ixs
    node_labels_ordered = copy(node_labels)
    for n, i in enumerate(indices):
        node_labels_ordered[i] = node_labels[n]
    
    return node_labels_ordered


def get_graph(agent):
    # defining a graph object based on the connectivity using networkx
    G = nx.from_numpy_matrix(agent.cm, create_using=nx.DiGraph())
    node_labels = get_node_labels(agent)
    mapping = {key:x for key,x in zip(range(agent.n_nodes),node_labels)}
    G = nx.relabel_nodes(G, mapping)
    return G




# Activity - Find unique states, transitions, transients
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def activity_list_causal_inputs(agent, df_activity = None):
    # input of t-1 with out and hidden after update (t)
    if type(df_activity) == type(None):
        df_activity = agent.activity 

    activity = []
    for row in range(len(df_activity)):
        state = [0 for i in range(agent.n_nodes)]
        
        s_input = df_activity.iloc[row]['input'].split(',')
        for c, i  in enumerate(agent.sensor_ixs):
            state[i] = int(s_input[c])
        
        s_output = df_activity.iloc[row]['output'].split(',')
        for c, i  in enumerate(agent.motor_ixs):
            state[i] = int(s_output[c])
        
        s_hidden = df_activity.iloc[row]['hiddenAfter'].split(',')
        for c, i  in enumerate(agent.hidden_ixs):
            state[i] = int(s_hidden[c])

        activity.append(state)

    return activity

def activity_list_concurrent_inputs(agent, df_activity = None):
    # input of t with out and hidden at t updated based on input at t-1
    # first state: motors set to 0
    # not that when the state is updated in MABE, motors are always set to 0 (thus there is no 
    # motorBefore, but motor = motorAfter)
    if type(df_activity) == type(None):
        df_activity = agent.activity 

    activity = []
    for row in range(len(df_activity)):
        state = [0 for i in range(agent.n_nodes)]
        
        s_input = df_activity.iloc[row]['input'].split(',')
        for c, i  in enumerate(agent.sensor_ixs):
            state[i] = int(s_input[c])
        
        # First state motors are 0, then current state motor is update from t-1
        if row > 0:
            s_output = df_activity.iloc[row-1]['output'].split(',')
            for c, i  in enumerate(agent.motor_ixs):
                state[i] = int(s_output[c])
        
        s_hidden = df_activity.iloc[row]['hiddenBefore'].split(',')
        for c, i  in enumerate(agent.hidden_ixs):
            state[i] = int(s_hidden[c])

        activity.append(state)

    return activity

def get_unique_states(agent, node_indices = None, return_counts = False):

    if type(node_indices) is not (np.ndarray or type(None)):
        node_indices = np.array(node_indices) 
    if type(node_indices) == type(None):
        node_indices = range(agent.n_nodes)

    # input of t-1 with out and hidden after update (t)
    activity = np.array(activity_list_concurrent_inputs(agent))

    return np.unique(activity[:,node_indices], return_counts = return_counts, axis = 0)


def get_unique_transitions(agent, trials= None, trial_col = 'life', return_counts = False, node_ind_pair = None, n_t = 1):
    # Note: this goes forwards in time with n_t
    '''
    Function for getting all unique transitions a system goes through in its lifetime.
    Transitions shouldn't cross trials, because there is no causal relationship between states across trials.
        Inputs:
            agent: agent object
            trial: the number of a specific trial to investigate (int, if None then all trials are considered)
            node_ind_pair: here two sets of indices for t and t+n_t
        Outputs:
            unique_states: an np.array of all unique transitions found 
    '''
   
    if type(node_ind_pair) == type(None):
        node_ind_pair [range(agent.n_nodes), range(agent.n_nodes)]

    if type(trials) == type(None):
        trials = pd.unique(agent.activity[trial_col])

    activity_array = []

    for t in trials:
        df_activity = agent.activity[agent.activity[trial_col] == t]
        activity = np.array(activity_list_concurrent_inputs(agent, df_activity = df_activity))

        activity_t = activity[:,np.array(node_ind_pair[0])]
        activity_nt = activity[:,np.array(node_ind_pair[1])]

        if n_t == 0:
            trial_trans = np.concatenate((activity_t, activity_nt), axis=1)    
        else: 
            trial_trans = np.concatenate((activity_t[0:-n_t], activity_nt[n_t:]), axis=1)
        activity_array.extend(trial_trans.tolist())
    
    return np.unique(np.array(activity_array), return_counts = return_counts, axis = 0)


def get_unique_multiple_ts(agent, trials = None, trial_col = 'life', return_counts = False, node_ind_array = None, n_t = 2):
    # Note: this goes backwards in time with n_t
    '''
    Function for getting all unique transitions a system goes through in its lifetime. For the second set of nodes
    in node_ind_pair, all n_t past states are considered
        Inputs:
            trial: the number of a specific trial to investigate (int, if None then all trials are considered)
            node_ind_array: sets of indices of size n_t+1, first set time n_t, second for n_t-1, etc., so ordered backwards in time
        Outputs:
            unique_states: an np.array of all unique transitions found 
    '''
    
    if type(node_ind_array) == type(None):
        return None

    if type(trials) is type(None):
        trials = pd.unique(agent.activity[trial_col])
        
    activity_array = []
    
    for t in trials:
        df_activity = agent.activity[agent.activity[trial_col] == t]
        activity = np.array(activity_list_concurrent_inputs(agent, df_activity = df_activity))

        activity_t = activity[:,np.array(node_ind_array[0])]
        trial_trans = activity_t[n_t:]
        
        for n in range(1,n_t+1):
            activity_nt = activity[:,np.array(node_ind_array[n])]
            trial_trans = np.concatenate((trial_trans, activity_nt[n_t-n:-n]), axis=1)
        activity_array.extend(trial_trans.tolist())

    return np.unique(np.array(activity_array), return_counts = return_counts, axis = 0)


# Activity - Probability Distributions
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _order_probabilities(probs, states, n_nodes):
    st_idx = [state2num(s) for s in states]
    st_array = np.zeros((2**n_nodes,))
    st_array[st_idx] = probs
    return st_array

def get_st_prob_distribution(agent, node_indices = None, order = False):
        '''
        Function for computing the probability distributions of sensor, motor, system states, and the joint distributions
        for one timestep
            Inputs: 
                node_indices: if None, all nodes are used
            Outputs:
                probability distribution, either nonzeros, or ordered in loli
        '''
        if node_indices == None:
            node_indices = range(agent.n_nodes)
        
        states, count = get_unique_states(agent, node_indices = node_indices, return_counts = True)
        probs = count/sum(count)

        # To compute entropies, leave out zeros, otherwise add and order states
        if order == True:
            probs = _order_probabilities(probs, states, len(node_indices))
        
        return probs

def get_trans_prob_distribution(agent, node_ind_pair = None, n_t = 1, order = False):
        '''
        Function for computing the probability distributions of sensor, motor, system states, and the joint distributions
        across n_t timesteps
            Inputs: 
                node_indices: if None, all nodes are used
            Outputs:
                probability distribution, either nonzeros, or ordered in loli
        '''

        if node_ind_pair == None:
            node_ind_pair = [range(agent.n_nodes), range(agent.n_nodes)]
        
        trans, count = get_unique_transitions(agent, node_ind_pair = node_ind_pair, return_counts = True, n_t = n_t)
        probs = count/sum(count)

        # To compute entropies, leave out zeros, otherwise add and order states
        if order == True:
            probs = _order_probabilities(probs, states, len(node_ind_pair[0]) + len(node_ind_pair[1]))
        
        return probs




# Convert
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Todo: Assumes binary, check with pyphi
def state2num(state,convention='loli'):
    '''
    Function description
        Inputs:
            inputs:
        Outputs:
            outputs:
    '''
    # send in a state as a list
    num = 0
    if convention == 'loli':
        for i in range(len(state)):
            num += (2**i)*state[i]
    else:
        for i in range(len(state)):
            num += (2**i)*state[-(i+1)]

    # returns the number associated with the state
    return int(num)

## Todo: Assumes binary
def num2state(num,n_nodes,convention='loli'):
    '''
    Function description
        Inputs:
            inputs:
        Outputs:
            outputs:
    '''

    number = '{0:0' + str(n_nodes) + 'b}'
    state = number.format(num)
    state = [int(i) for i in state]

    # returns the state
    if convention == 'loli':
        state.reverse()
        return state
    else:
        return state



# Not tested 6/18/2021 ----------
def get_state_tuple(agent, trial, t):
        '''
        Function for picking out a specific state of the system.
         Inputs:
             trial: the trial number that is under investigation (int)
             t: the timestep you wish to find the transition to (int)
         Outputs:
             two tuples (X and Y in Albantakis et al 2019) containing the state of the system at time t-1 and t.
        '''
        # Checking if brain activity exists
        if not hasattr(agent, 'brain_activity'):
            raise AttributeError('No brain activity saved yet.')

        # return state as a tuple (can be used for calculating Phi)
        return tuple(agent.brain_activity[trial, t].astype(int))


# OLD CHECK
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# TODO: For Sensors I actually only have to exclude the last, for motors only the first.
def get_unique_states_old(brain_activity, trial= None, exclude_first_last = True, node_indices = None, return_counts = False):
    '''
    Function for getting all unique states a system goes through in its lifetime.
        Inputs:
            brain_activity: organized as n_trials x n_times
            trial: the number of a specific trial to investigate (int, if None then all trials are considered)
            exclude_first_last: if activity is from MABE the first and last state should be excluded
        Outputs:
            unique_states: an np.array of all unique states found
    '''
    n_trials = brain_activity.shape[0]
    n_times = brain_activity.shape[1]
    n_nodes = brain_activity.shape[2]

    if node_indices == None:
        node_indices = range(n_nodes)

    if trial is None: 
        activity_array = []
        if exclude_first_last:
            for t in brain_activity:
                activity_array.extend(t[1:-1, node_indices].tolist())
        else:
            for t in brain_activity:
                activity_array.extend(t[:, node_indices])
    else: 
        activity_array = brain_activity[trial][:,node_indices]
        if exclude_first_last:
            activity_array = activity_array[1:-1]

    return np.unique(np.array(activity_array), return_counts = return_counts, axis = 0)

# Todo: make another function in pyAgent, that can 
# trim_sen_mot: cut motors from before_state and sensors from after_state
# This would be to speed up the AC analysis, and possibly the information analysis
def get_unique_transitions_old(brain_activity, trial= None, return_counts = False, node_ind_pair = None, n_t = 1):
    '''
    Function for getting all unique transitions a system goes through in its lifetime.
        Inputs:
            brain_activity: organized as n_trials x n_times
            trial: the number of a specific trial to investigate (int, if None then all trials are considered)
            node_ind_pair: here two sets of indices for t and t+1
        Outputs:
            unique_states: an np.array of all unique transitions found 
    '''
    n_trials = brain_activity.shape[0]
    n_times = brain_activity.shape[1]
    n_nodes = brain_activity.shape[2]
    
    if node_ind_pair == None:
        node_ind_pair = [range(n_nodes), range(n_nodes)]

    if trial is None: 
        activity_array = []
        for t in brain_activity:
            trial_trans = np.concatenate((t[0:-n_t, node_ind_pair[0]], t[n_t:, node_ind_pair[1]]), axis=1)
            activity_array.extend(trial_trans.tolist())
    else: 
        t = brain_activity[trial]
        activity_array = np.concatenate((t[0:-n_t, node_ind_pair[0]], t[n_t:, node_ind_pair[1]]), axis=1)

    return np.unique(np.array(activity_array), return_counts = return_counts, axis = 0)

def get_unique_multiple_ts_old(brain_activity, trial= None, return_counts = False, node_ind_array = None, n_t = 2):
    '''
    Function for getting all unique transitions a system goes through in its lifetime. For the second set of nodes
    in node_ind_pair, all n_t past states are considered
        Inputs:
            brain_activity: organized as n_trials x n_times
            trial: the number of a specific trial to investigate (int, if None then all trials are considered)
            node_ind_array: sets of indices of size n_t+1, first set time n_t, second for n_t-1, etc., so ordered backwards in time
        Outputs:
            unique_states: an np.array of all unique transitions found 
    '''
    n_trials = brain_activity.shape[0]
    n_times = brain_activity.shape[1]
    
    if node_ind_array == None:
        return None

    if trial is None:
        activity_array = []
        for t in brain_activity:
            trial_trans = t[n_t:, node_ind_array[0]]
            for n in range(1,n_t+1):
                trial_trans = np.concatenate((trial_trans, t[n_t-n:-n, node_ind_array[n]]), axis=1)
            activity_array.extend(trial_trans.tolist())
    else: 
        t = brain_activity[trial]
        trial_trans = t[n_t:, node_ind_array[0]]
        for n in range(1,n_t+1):
            trial_trans = np.concatenate((trial_trans, t[n_t-n:-n, node_ind_array[n]]), axis=1)
        activity_array = trial_trans

    return np.unique(np.array(activity_array), return_counts = return_counts, axis = 0)


def get_st_prob_distribution_old(agent, exclude_first_last = True, node_indices = None, order = False):
        '''
        Function for computing the probability distributions of sensor, motor, system states, and the joint distributions
        for on timestep
            Inputs: 
                node_indices: if None, all nodes are used
            Outputs:
                unique_states: an np.array of all unique transitions found 
        '''
        if node_indices == None:
            node_index = range(agent.n_nodes)
        
        states, count = get_unique_states(agent.brain_activity, node_indices = node_indices, exclude_first_last = True, return_counts = True)
        probs = count/sum(count)

        if order == False:
            # This is to compute entropies
            return probs
        else:
            st_idx = [state2num(s) for s in states]
            st_array = np.zeros((2**(len(node_indices)),))
            st_array[st_idx] = probs
            return st_array


def get_trans_prob_distribution_old(agent, node_ind_pair = None, n_t = 1, order = False):
        '''
        Function for computing the probability distributions of sensor, motor, system states, and the joint distributions
        across n_t timesteps
            Inputs: 
                node_indices: if None, all nodes are used
            Outputs:
                unique_states: an np.array of all unique transitions found 
        '''

        if node_ind_pair == None:
            node_ind_pair = [range(agent.n_nodes), range(agent.n_nodes)]
        
        trans, count = get_unique_transitions(agent.brain_activity, node_ind_pair = node_ind_pair, return_counts = True, n_t = n_t)
        probs = count/sum(count)

        if order == False:
            # This is to compute entropies
            return probs
        else:
            st_idx = [state2num(s) for s in trans]
            st_array = np.zeros((2**(len(node_ind_pair[0]) + len(node_ind_pair[1])),))
            st_array[st_idx] = probs
            return st_array

