# causal_agent_analysis.py

import itertools

import numpy as np
import pyphi
from scipy.stats import entropy

from .shapley_values import compute_shapley_values
from .utils import *

###############################################################################
# Collection of functions to assess the causal properties of an agent based on
# its transition probability matrix
###############################################################################

# specific utils
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def fix_TPM_dim(agent, motors=False):
    # PathFollow agents TPMs are only specified for motors = 000 to save memory,
    # motors do not feed back
    if motors is False:
        ind = np.sort(agent.sensor_ixs + agent.hidden_ixs)
        cm = np.array(agent.cm)[:, ind][ind]
        if len(agent.tpm) is not (2 ** agent.n_nodes):
            tpm = np.array(agent.tpm)[:, ind]
        else:
            net = pyphi.Network(agent.tpm)
            tpm = pyphi.tpm.marginalize_out(agent.motor_ixs, net.tpm)
            tpm = pyphi.convert.to_2dimensional(tpm)
            tpm = np.array(tpm)[:, ind]
    else:
        ind = None
        cm = agent.cm
        if len(agent.tpm) is not (2 ** agent.n_nodes):
            tpm = np.tile(np.array(agent.tpm), (2 ** agent.n_motors, 1))
        else:
            tpm = agent.tpm

    return tpm, cm, ind


# Entropies
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def sys_entropy(agent):
    # Just seems to be the effective info plus the entropy of the sensors, which
    # is max ent in the TPM
    tpm, _, _ = fix_TPM_dim(agent, motors=True)
    sbs_tpm = pyphi.convert.state_by_node2state_by_state(tpm)
    avg_repertoire = np.mean(sbs_tpm, 0)
    return entropy(avg_repertoire, base=2.0)


# TODO: fix
# def node_entropy(network):
# 	avg_prob = np.mean(network.tpm.reshape([number_of_states]+[number_of_nodes], order = 'F'),0)

# 	return np.mean([entropy([p, 1-p], base = 2.) for p in avg_prob])

# Autonomy Causal
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def effective_information(agent):
    # TODO: A_m(n_t = 1) is the same as the effective info and computes much faster
    tpm, _, _ = fix_TPM_dim(agent, motors=True)
    sbs_tpm = pyphi.convert.state_by_node2state_by_state(tpm)
    avg_repertoire = np.mean(sbs_tpm, 0)

    return np.mean([entropy(repertoire, avg_repertoire, 2.0) for repertoire in sbs_tpm])


def A_m_sensors_causal(agent, n_t=5):
    # H(OM_t|S^_t-1,...S^_t-n) - H(OM_t|OM^_t-1, S^_t-1,...S^_t-n)
    # Note that for deterministic systems the second part is 0
    # --> A^_m = H(OM_t|S^_t-1, ..., S^_t-m)
    # TODO: add the part for non-deterministic agents (vvv)

    tpm, _, _ = fix_TPM_dim(agent, motors=True)
    network = pyphi.Network(tpm)

    ind_hm = tuple(agent.hidden_ixs + agent.motor_ixs)
    ind_s = tuple(agent.sensor_ixs)

    num_input_states = 2 ** agent.n_sensors

    # make conditional tpms for every possible input state
    tpm_cond_input = []
    for si in range(num_input_states):
        sensor_state = num2state(si, agent.n_sensors)

        # condition TPM (hidden and motor state doesn't matter, but needed for conditioning function)
        full_state = np.array([0 for i in range(agent.n_nodes)])
        full_state[agent.sensor_ixs] = sensor_state

        tpm_cond = pyphi.tpm.condition_tpm(
            pyphi.convert.to_multidimensional(tpm), agent.sensor_ixs, full_state
        )
        tpm_cond = pyphi.convert.to_2dimensional(tpm_cond)
        tpm_cond = np.array(tpm_cond)[:, ind_hm]
        tpm_cond = pyphi.convert.state_by_node2state_by_state(tpm_cond)

        tpm_cond_input.append(tpm_cond)

    input_combinations = [
        p for p in itertools.product(range(num_input_states), repeat=n_t)
    ]

    av_sequence_rep = []
    for ic in input_combinations:
        # start with max ent
        maxent_val = 1 / (2 ** len(ind_hm))
        avg_repertoire = [maxent_val for s in range(2 ** len(ind_hm))]
        for ind_in_state in ic:

            in_tpm = tpm_cond_input[ind_in_state]

            # get average repertoire
            avg_repertoire = np.average(in_tpm, weights=avg_repertoire, axis=0)

        av_sequence_rep.append(avg_repertoire)

    return entropy(np.mean(av_sequence_rep, 0), base=2)


# Actual Causation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def alpha_ratio_hidden(agent):
    tpm, cm, _ = fix_TPM_dim(agent, motors=True)

    # TODO: once it is possible to define node specific transitions in pyphi, improve
    ind_hs = tuple(agent.hidden_ixs + agent.sensor_ixs)
    ind_m = tuple(agent.motor_ixs)
    # node_ind_pair = [ind_hs, ind_m]

    transitions, counts = get_unique_transitions(
        agent, return_counts=True, node_ind_pair=None, n_t=1
    )
    probs = counts / sum(counts)

    network = pyphi.Network(tpm, cm=cm)

    hidden_alpha_ratio = []
    for t, trans in enumerate(transitions):
        trans = np.array(trans)
        transition = pyphi.Transition(
            network,
            trans[range(0, agent.n_nodes)],
            trans[range(agent.n_nodes, 2 * agent.n_nodes)],
            ind_hs,
            ind_m,
        )

        account = pyphi.actual.account(
            transition, direction=pyphi.direction.Direction.CAUSE
        )

        if not account:
            hidden_alpha_ratio.append(0)
        else:
            sum_alpha = sum([d.alpha for d in account])

            sum_alpha_hidden = 0
            for d in account:
                # extended purviews outputs all tied actual causes
                purviews = d.extended_purview
                sum_alpha_hidden_purview = 0

                for pur in purviews:
                    hidden_purview_idx = np.array(
                        [c for c, p in enumerate(pur) if p in agent.hidden_ixs]
                    )

                    if len(hidden_purview_idx) == len(pur):
                        # if all purview nodes are hidden, add total alpha
                        sum_alpha_hidden_purview = sum_alpha_hidden_purview + d.alpha

                    elif len(hidden_purview_idx) > 0:
                        # TODO: improve, only compute for the hidden nodes
                        shapley_values = np.array(
                            compute_shapley_values(d, transition, pur)
                        )
                        sum_alpha_hidden_purview = sum_alpha_hidden_purview + sum(
                            shapley_values[hidden_purview_idx]
                        )

                # If there are multiple equivalent purviews, get average hidden contribution
                sum_alpha_hidden_purview = sum_alpha_hidden_purview / len(purviews)

                sum_alpha_hidden = sum_alpha_hidden + sum_alpha_hidden_purview
         
            hidden_alpha_ratio.append(sum_alpha_hidden / sum_alpha)
    
    average_hidden_alpha_ratio = sum(hidden_alpha_ratio * probs)

    return average_hidden_alpha_ratio


def transition_alpha_ratio(agent):
    tpm, cm, _ = fix_TPM_dim(agent, motors=True)

    ind_hs = tuple(agent.hidden_ixs + agent.sensor_ixs)
    ind_m = tuple(agent.motor_ixs)

    transitions, _ = get_unique_transitions(
        agent, return_counts=True, node_ind_pair=None, n_t=1
    )

    network = pyphi.Network(tpm, cm=cm)

    alpha_ratio_transition = []
    for _, trans in enumerate(transitions):
        trans = np.array(trans)
        transition = pyphi.Transition(
            network,
            trans[:agent.n_nodes],
            trans[agent.n_nodes:2 * agent.n_nodes],
            ind_hs,
            ind_m,
        )

        account = pyphi.actual.account(
            transition, direction=pyphi.direction.Direction.CAUSE
        )

        if not account:
            alpha_ratio_transition.append(np.zeros(len(ind_hs)))
        else:
            sum_alpha = sum([d.alpha for d in account])

            alpha_ratio = np.zeros(len(ind_hs))
            for d in account:
                # extended purviews outputs all tied actual causes
                purviews = d.extended_purview
                alpha_purview = np.zeros(len(ind_hs))

                for pur in purviews:

                    for c, p in enumerate(pur):
                        purview_idx = np.array([p])
                    
                        if len(purview_idx) == len(pur):
                            alpha_purview[p] += d.alpha

                        elif len(purview_idx) > 0:

                            shapley_values = np.array(
                                compute_shapley_values(d, transition, pur)
                            )
                            alpha_purview[p] += shapley_values[c]
                           
                # If there are multiple equivalent purviews, get average hidden contribution
                alpha_purview = alpha_purview / len(purviews)

                alpha_ratio += alpha_purview
         
            alpha_ratio_transition.append(alpha_ratio / sum_alpha)
    
    return alpha_ratio_transition


# IIT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def sum_of_small_phi_full_system(agent, save_agent=False):

    tpm, cm, ind = fix_TPM_dim(agent, motors=True)

    states, counts = get_unique_states(
        agent, node_indices=ind, return_counts=True, causal=True
    )
    probs = counts / sum(counts)

    if save_agent:
        agent.sum_phi_full_3_0 = 0.0

    network = pyphi.Network(tpm, cm=cm)

    sum_phi = []
    for s, state in enumerate(states):

        subsystem = pyphi.Subsystem(network, state)
        ces = pyphi.compute.ces(subsystem)

        sum_phi_CES = sum([d.phi for d in ces])
        sum_phi.append(sum_phi_CES)

    average_sum_phi = sum(sum_phi * probs)
    if save_agent:
        agent.sum_phi_full_3_0 = average_sum_phi

    return average_sum_phi


def average_IIT_3_0(agent, save_agent=False):

    tpm, cm, ind = fix_TPM_dim(agent, motors=False)

    states, counts = get_unique_states(
        agent, node_indices=ind, return_counts=True, causal=True
    )
    probs = counts / sum(counts)

    if save_agent:
        agent.Phi_3_0 = {"states": {}, "average_Phi": 0.0, "average_sum_phi": 0.0}

    network = pyphi.Network(tpm, cm=cm)

    Phi = []
    sum_phi = []
    for s, state in enumerate(states):
        sia = pyphi.compute.major_complex(network, state)
        Phi.append(sia.phi)

        sum_phi_CES = sum([d.phi for d in sia.ces])
        sum_phi.append(sum_phi_CES)

        if save_agent:
            agent.Phi_3_0["states"][s] = {
                "major_complex": sia.subsystem.node_indices,
                "state": state,
                "Phi": sia.phi,
                "num_distinctions": len(sia.ces),
                "sum_phi": sum_phi_CES,
            }

    average_Phi = sum(Phi * probs)
    average_sum_phi = sum(sum_phi * probs)
    if save_agent:
        agent.Phi_3_0["average_Phi"] = average_Phi
        agent.Phi_3_0["average_sum_phi"] = average_sum_phi

    return average_Phi, average_sum_phi


# All
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def fullCausalAnalysis(agent, save_agent=False):

    average_Phi, average_sum_phi_MC = average_IIT_3_0(agent)

    causal_measures = {
        "E": effective_information(agent),
        "A_4c": A_m_sensors_causal(agent, n_t=4),
        "Phi_max": average_Phi,
        "sum_phi_MC": average_sum_phi_MC,
        "sum_phi_full": sum_of_small_phi_full_system(agent),
        "alpha_ratio_hidden": alpha_ratio_hidden(agent),
    }

    df = pd.DataFrame(causal_measures, index=[1])

    if save_agent:
        agent.causal_analysis = df

    return df


def emptyCausalAnalysis(index_num=1):
    df = pd.DataFrame(
        dtype=float,
        columns=[
            "E",
            "A_4c",
            "Phi_max",
            "sum_phi_MC",
            "sum_phi_full",
            "alpha_ratio_hidden",
        ],
        index=range(index_num),
    )
    return df
