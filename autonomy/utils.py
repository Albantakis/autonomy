from copy import copy

import networkx as nx
import numpy as np
import pandas as pd


# General
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_node_labels(agent):
    node_labels = []

    # standard labels for up to 10 nodes of each kind
    sensor_labels = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"]
    motor_labels = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10"]
    hidden_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    # defining labels for each node type
    s = [sensor_labels[i] for i in list(range(agent.n_sensors))]
    m = [motor_labels[i] for i in list(range(agent.n_motors))]
    h = [hidden_labels[i] for i in list(range(agent.n_hidden))]

    # combining the labels
    node_labels.extend(s)
    node_labels.extend(m)
    node_labels.extend(h)

    # sort node labels according to idx
    indices = agent.sensor_ixs + agent.motor_ixs + agent.hidden_ixs
    node_labels_ordered = copy(node_labels)
    for n, i in enumerate(indices):
        node_labels_ordered[i] = node_labels[n]

    return node_labels_ordered


def get_graph(agent):
    # defining a graph object based on the connectivity using networkx
    G = nx.from_numpy_matrix(np.array(agent.cm), create_using=nx.DiGraph())
    node_labels = get_node_labels(agent)
    mapping = {key: x for key, x in zip(range(agent.n_nodes), node_labels)}
    G = nx.relabel_nodes(G, mapping)
    return G


def append_to_df(agent, df_type, df_function, df_function_inputs, df_colname):

    df_type_fullname = {
        "S": "structural_analysis",
        "I": "info_analysis",
        "C": "causal_analysis",
        "D": "dynamical_analysis",
    }[df_type]

    df = getattr(agent, df_type_fullname)

    df[df_colname] = df_function(*df_function_inputs)


# Activity - Find unique states, transitions, transients
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def activity_list_causal_inputs(agent, df_activity=None):
    # input of t-1 with out and hidden after update (t)
    if type(df_activity) == type(None):
        df_activity = agent.activity

    activity = []
    for row in range(len(df_activity)):
        state = [0 for i in range(agent.n_nodes)]

        if isinstance(df_activity.iloc[row]["input"], str):

            s_input = df_activity.iloc[row]["input"].split(",")
            for c, i in enumerate(agent.sensor_ixs):
                state[i] = int(s_input[c])

            s_output = df_activity.iloc[row]["output"].split(",")
            for c, i in enumerate(agent.motor_ixs):
                state[i] = int(s_output[c])

            s_hidden = df_activity.iloc[row]["hiddenAfter"].split(",")
            for c, i in enumerate(agent.hidden_ixs):
                state[i] = int(s_hidden[c])

        else:
            s_input = df_activity.iloc[row]["input"]
            for c, i in enumerate(agent.sensor_ixs):
                state[i] = s_input[c]

            s_output = df_activity.iloc[row]["output"]
            for c, i in enumerate(agent.motor_ixs):
                state[i] = s_output[c]

            s_hidden = df_activity.iloc[row]["hiddenAfter"]
            for c, i in enumerate(agent.hidden_ixs):
                state[i] = s_hidden[c]

        activity.append(state)

    return activity


def activity_list_concurrent_inputs(agent, df_activity=None):
    # input of t with out and hidden at t updated based on input at t-1
    # first state: motors set to 0
    # not that when the state is updated in MABE, motors are always set to 0 (thus there is no
    # motorBefore, but motor = motorAfter)
    if type(df_activity) == type(None):
        df_activity = agent.activity

    activity = []
    for row in range(len(df_activity)):
        state = [0 for i in range(agent.n_nodes)]

        if isinstance(df_activity.iloc[row]["input"], str):

            s_input = df_activity.iloc[row]["input"].split(",")
            for c, i in enumerate(agent.sensor_ixs):
                state[i] = int(s_input[c])

            # First state motors are 0, then current state motor is update from t-1
            if row > 0:
                s_output = df_activity.iloc[row - 1]["output"].split(",")
                for c, i in enumerate(agent.motor_ixs):
                    state[i] = int(s_output[c])

            s_hidden = df_activity.iloc[row]["hiddenBefore"].split(",")
            for c, i in enumerate(agent.hidden_ixs):
                state[i] = int(s_hidden[c])
        else:
            s_input = df_activity.iloc[row]["input"]
            for c, i in enumerate(agent.sensor_ixs):
                state[i] = s_input[c]

            # First state motors are 0, then current state motor is update from t-1
            if row > 0:
                s_output = df_activity.iloc[row - 1]["output"]
                for c, i in enumerate(agent.motor_ixs):
                    state[i] = s_output[c]

            s_hidden = df_activity.iloc[row]["hiddenBefore"]
            for c, i in enumerate(agent.hidden_ixs):
                state[i] = s_hidden[c]

        activity.append(state)

    return activity


def get_unique_states(agent, node_indices=None, return_counts=False, causal=False):

    if type(node_indices) == type(None):
        node_indices = range(agent.n_nodes)

    elif type(node_indices) is not type(np.ndarray):
        node_indices = np.array(node_indices)

    if causal == True:
        # input of t-1 with out and hidden after update (t)
        activity = np.array(activity_list_causal_inputs(agent))
    else:
        # input, hidden, output all at same t. First ts, motors set to 0
        activity = np.array(activity_list_concurrent_inputs(agent))

    return np.unique(activity[:, node_indices], return_counts=return_counts, axis=0)


def get_unique_transitions(
    agent, trials=None, trial_col="life", return_counts=False, node_ind_pair=None, n_t=1
):
    # Note: this goes forwards in time with n_t
    """
    Function for getting all unique transitions a system goes through in its lifetime.
    Transitions shouldn't cross trials, because there is no causal relationship between states across trials.
        Inputs:
            agent: agent object
            trial: the number of a specific trial to investigate (int, if None then all trials are considered)
            node_ind_pair: here two sets of indices for t and t+n_t
        Outputs:
            unique_states: an np.array of all unique transitions found
    """

    if type(node_ind_pair) == type(None):
        node_ind_pair = [range(agent.n_nodes), range(agent.n_nodes)]

    if type(trials) == type(None):
        trials = pd.unique(agent.activity[trial_col])

    activity_array = []

    for t in trials:
        df_activity = agent.activity[agent.activity[trial_col] == t]
        activity = np.array(
            activity_list_concurrent_inputs(agent, df_activity=df_activity)
        )

        activity_t = activity[:, np.array(node_ind_pair[0])]
        activity_nt = activity[:, np.array(node_ind_pair[1])]

        if n_t == 0:
            trial_trans = np.concatenate((activity_t, activity_nt), axis=1)
        else:
            trial_trans = np.concatenate(
                (activity_t[0:-n_t], activity_nt[n_t:]), axis=1
            )
        activity_array.extend(trial_trans.tolist())

    return np.unique(np.array(activity_array), return_counts=return_counts, axis=0)


def get_unique_multiple_ts(
    agent,
    trials=None,
    trial_col="life",
    return_counts=False,
    node_ind_array=None,
    n_t=2,
):
    # Note: this goes backwards in time with n_t
    """
    Function for getting all unique transitions a system goes through in its lifetime. For the second set of nodes
    in node_ind_pair, all n_t past states are considered
        Inputs:
            trial: the number of a specific trial to investigate (int, if None then all trials are considered)
            node_ind_array: sets of indices of size n_t+1, first set time n_t, second for n_t-1, etc., so ordered backwards in time
        Outputs:
            unique_states: an np.array of all unique transitions found
    """

    if type(node_ind_array) == type(None):
        return None

    if type(trials) is type(None):
        trials = pd.unique(agent.activity[trial_col])

    activity_array = []

    for t in trials:
        df_activity = agent.activity[agent.activity[trial_col] == t]
        activity = np.array(
            activity_list_concurrent_inputs(agent, df_activity=df_activity)
        )

        activity_t = activity[:, np.array(node_ind_array[0])]
        trial_trans = activity_t[n_t:]

        for n in range(1, n_t + 1):
            activity_nt = activity[:, np.array(node_ind_array[n])]
            trial_trans = np.concatenate(
                (trial_trans, activity_nt[n_t - n : -n]), axis=1
            )
        activity_array.extend(trial_trans.tolist())

    return np.unique(np.array(activity_array), return_counts=return_counts, axis=0)


# Activity - Probability Distributions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def _order_probabilities(probs, states, n_nodes):
    st_idx = [state2num(s) for s in states]
    st_array = np.zeros((2 ** n_nodes,))
    st_array[st_idx] = probs
    return st_array


def get_st_prob_distribution(agent, node_indices=None, order=False):
    """
    Function for computing the probability distributions of sensor, motor, system states, and the joint distributions
    for one timestep
        Inputs:
            node_indices: if None, all nodes are used
        Outputs:
            probability distribution, either nonzeros, or ordered in loli
    """
    if node_indices == None:
        node_indices = range(agent.n_nodes)

    states, count = get_unique_states(
        agent, node_indices=node_indices, return_counts=True
    )
    probs = count / sum(count)

    # To compute entropies, leave out zeros, otherwise add and order states
    if order == True:
        probs = _order_probabilities(probs, states, len(node_indices))

    return probs


def get_trans_prob_distribution(agent, node_ind_pair=None, n_t=1, order=False):
    """
    Function for computing the probability distributions of sensor, motor, system states, and the joint distributions
    across n_t timesteps
        Inputs:
            node_indices: if None, all nodes are used
        Outputs:
            probability distribution, either nonzeros, or ordered in loli
    """

    if node_ind_pair == None:
        node_ind_pair = [range(agent.n_nodes), range(agent.n_nodes)]

    trans, count = get_unique_transitions(
        agent, node_ind_pair=node_ind_pair, return_counts=True, n_t=n_t
    )
    probs = count / sum(count)

    # To compute entropies, leave out zeros, otherwise add and order states
    if order == True:
        probs = _order_probabilities(
            probs, states, len(node_ind_pair[0]) + len(node_ind_pair[1])
        )

    return probs


# Convert
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Todo: Assumes binary, check with pyphi
def state2num(state, convention="loli"):
    """
    Function description
        Inputs:
            inputs:
        Outputs:
            outputs:
    """
    # send in a state as a list
    num = 0
    if convention == "loli":
        for i in range(len(state)):
            num += (2 ** i) * state[i]
    else:
        for i in range(len(state)):
            num += (2 ** i) * state[-(i + 1)]

    # returns the number associated with the state
    return int(num)


## Todo: Assumes binary
def num2state(num, n_nodes, convention="loli"):
    """
    Function description
        Inputs:
            inputs:
        Outputs:
            outputs:
    """

    number = "{0:0" + str(n_nodes) + "b}"
    state = number.format(num)
    state = [int(i) for i in state]

    # returns the state
    if convention == "loli":
        state.reverse()
        return state
    else:
        return state
