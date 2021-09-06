import copy

import networkx as nx
import numpy as np
import pandas as pd
import pyphi
from networkx.algorithms import (
    betweenness_centrality,
    degree_centrality,
    flow_hierarchy,
)

from .utils import get_graph

#####################################################################################################################
### Collection of functions to assess the structural properties of an agent based on its connectivity matrix (cm) ###
#####################################################################################################################


def number_of_connected_sensors(cm, sensor_ixs):
    # Sensors with outputs
    # cm should be np.array
    return np.sum(np.sum(cm[sensor_ixs, :], 1) > 0)


def number_of_connected_motors(cm, motor_ixs):
    # Motors with inputs
    # cm should be np.array
    return np.sum(np.sum(cm[:, motor_ixs], 0) > 0)


def number_of_densely_connected_nodes(cm_agent, allow_self_loops=False):
    # num hidden nodes with inputs and outputs
    cm = np.array(copy.copy(cm_agent))
    if not allow_self_loops:
        for i in range(len(cm)):
            cm[i, i] = 0
    return np.sum((np.sum(cm, 0) * np.sum(cm, 1)) > 0)


def connected_nodes(agent):
    # Sensors with outputs, motors with inputs, and hidden with both
    cm = np.array(copy.copy(agent.cm))
    # kill self loops
    for i in range(len(cm)):
        cm[i, i] = 0

    S = np.array(agent.sensor_ixs)
    cS_ind = np.where(np.sum(cm[S, :], 1) > 0)[0]
    cS = list(S[cS_ind])

    M = np.array(agent.motor_ixs)
    cM_ind = np.where(np.sum(cm[:, M], 0) > 0)[0]
    cM = list(M[cM_ind])

    cH = list(densely_connected_nodes(cm))
    return np.sort(cS + cM + cH)


def number_of_connected_nodes_by_type(agent):
    cm = np.array(agent.cm)
    cS = number_of_connected_sensors(cm, agent.sensor_ixs)
    cH = number_of_densely_connected_nodes(cm)
    cM = number_of_connected_motors(cm, agent.motor_ixs)
    num_connected_nodes = {"cN": sum([cS, cH, cM]), "cS": cS, "cH": cH, "cM": cM}
    return pd.DataFrame(num_connected_nodes, index=[1])


def densely_connected_nodes(cm_agent, allow_self_loops=False):
    # Hidden nodes with inputs and outputs
    cm = copy.copy(cm_agent)
    if not allow_self_loops:
        for i in range(len(cm)):
            cm[i, i] = 0
    return np.where((np.sum(cm, 0) * np.sum(cm, 1)) > 0)[0]


def number_of_connections(cm, a_ixs, b_ixs):
    return np.sum(cm[np.ix_(a_ixs, b_ixs)] > 0)


def number_of_connections_by_type(agent, connected_only=True):
    cm = np.array(agent.cm)
    if connected_only:
        ind = set(connected_nodes(agent))
    else:
        ind = set(range(agent.n_nodes))

    S = list(ind.intersection(agent.sensor_ixs))
    M = list(ind.intersection(agent.motor_ixs))
    H = list(ind.intersection(agent.hidden_ixs))

    num_connections = {
        "s_m": number_of_connections(cm, S, M),
        "s_h": number_of_connections(cm, S, H),
        "h_h": number_of_connections(cm, H, H),
        "h_m": number_of_connections(cm, H, M),
    }
    return pd.DataFrame(num_connections, index=[1])


def LSCC(G):
    # largest strongly connected component using networkx graph
    LSCC = max(nx.strongly_connected_components(G), key=len)
    if len(LSCC) < 2:
        return None
    else:
        return LSCC


def len_LSCC(G):
    if len(G) > 0:
        LSCC = max(nx.strongly_connected_components(G), key=len)
        len_LSCC = len(LSCC)
    else:
        len_LSCC = 0
    return len_LSCC


def len_LWCC(G):
    if len(G) > 0:
        LWCC = max(nx.weakly_connected_components(G), key=len)
        len_LWCC = len(LWCC)
    else:
        len_LWCC = 0
    return len_LWCC


def average_betweenness_centrality(G, connected_only=True):
    # Betweenness centrality of a node v is the sum of the fraction of all-pairs
    # shortest paths that pass through v.
    # Only densely connected hidden nodes can have positive betweenness_centrality
    HBC = betweenness_centrality(G)
    if connected_only:
        cm = np.array(nx.adjacency_matrix(G).todense())
        num_hidden = number_of_densely_connected_nodes(cm)
    else:
        num_hidden = len(agent.hidden_ixs)

    avHBC = sum([HBC[b] for b in HBC]) / num_hidden
    return avHBC


def average_degree_centrality(G, connected_only=True):
    # The degree centrality for a node v is the fraction of nodes it is connected to.
    DC = degree_centrality(G)
    DC_list = [DC[d] for d in DC]
    avDC = sum(DC_list) / len(DC_list)
    return avDC


# All
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def fullStructuralAnalysis(agent, connected_only=True, save_agent=False):
    df = number_of_connected_nodes_by_type(agent)
    df = df.join(number_of_connections_by_type(agent, connected_only=connected_only))

    cm = np.array(agent.cm)
    # Components
    if connected_only == True:
        ind_con = connected_nodes(agent)
        if len(ind_con) > 0:
            cm_connected = cm[np.ix_(ind_con, ind_con)]
            G = nx.from_numpy_matrix(cm_connected, create_using=nx.DiGraph())
        else:
            G = nx.empty_graph(n=0, create_using=nx.DiGraph())
    else:
        G = get_graph(agent)

    if (G.size() > 0) and (len(G) > 0):  # G.size is number of edges
        components = {
            "len_LSCC": len_LSCC(G),
            "len_LWCC": len_LWCC(G),
            "flow_hierarchy": flow_hierarchy(
                G
            ),  # Flow hierarchy is defined as the fraction of edges not participating in cycles in a directed graph
            "av_betweenness_centrality": average_betweenness_centrality(
                G, connected_only
            ),
            "av_degree_centrality": average_degree_centrality(G),
        }
    else:
        components = {
            "len_LSCC": 0,
            "len_LWCC": 0,
            "flow_hierarchy": 1.0,
            "av_betweenness_centrality": 0.0,
            "av_degree_centrality": 0.0,
        }

    df = df.join(pd.DataFrame(components, index=[1]))

    if save_agent:
        agent.structural_analysis = df

    return df


def emptyStructuralAnalysis(index_num=1):
    df = pd.DataFrame(
        dtype=float,
        columns=[
            "cN",
            "cS",
            "cH",
            "cM",
            "s_m",
            "s_h",
            "h_h",
            "h_m",
            "len_LSCC",
            "len_LWCC",
            "flow_hierarchy",
            "av_betweenness_centrality",
            "av_degree_centrality",
        ],
        index=range(index_num),
    )
    return df
