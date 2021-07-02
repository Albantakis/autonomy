
import numpy as np
import pyphi
import networkx as nx
import copy
import pandas as pd
from utils import get_graph
from networkx.algorithms import degree_centrality, betweenness_centrality, flow_hierarchy

#####################################################################################################################
### Collection of functions to assess the structural properties of an agent based on its connectivity matrix (cm) ###

# Todo: 
# - Number of connections includes all nodes, not just connected nodes. Make option.
# - spectral density (Banerjee and Jost, 2009)? --> Too complicated.
# - density of connections and weight distributions (the latter for ANNs)

#####################################################################################################################

def number_of_connected_sensors(cm,sensor_ixs):
    # Sensors with outputs
    return np.sum(np.sum(cm[sensor_ixs,:],1)>0)

def number_of_connected_motors(cm,motor_ixs):
    # Motors with inputs
    return np.sum(np.sum(cm[:,motor_ixs],0)>0)

def number_of_densely_connected_nodes(cm_agent,allow_self_loops=False):
    # num hidden nodes with inputs and outputs
    cm = copy.copy(cm_agent)
    if not allow_self_loops:
        for i in range(len(cm)):
            cm[i,i] = 0
    return np.sum((np.sum(cm,0)*np.sum(cm,1))>0)

def number_of_connected_nodes_by_type(agent):
    cm = agent.cm
    cS = number_of_connected_sensors(cm,agent.sensor_ixs)
    cH = number_of_densely_connected_nodes(cm)
    cM = number_of_connected_motors(cm,agent.motor_ixs)
    num_connected_nodes = {
        'cN': sum([cS, cH, cM]),
        'cS': cS,
        'cH': cH,
        'cM': cM
        }
    return pd.DataFrame(num_connected_nodes, index = [1])

def densely_connected_nodes(cm_agent,allow_self_loops=False):
    # Hidden nodes with inputs and outputs
    cm = copy.copy(cm_agent)
    if not allow_self_loops:
        for i in range(len(cm)):
            cm[i,i] = 0
    return np.where((np.sum(cm,0)*np.sum(cm,1))>0)[0]

def number_of_connections(cm,a_ixs,b_ixs):
    return np.sum(cm[np.ix_(a_ixs,b_ixs)]>0)

def number_of_connections_by_type(agent):
    cm = agent.cm
    num_connections = {
        's_m': number_of_connections(cm,agent.sensor_ixs, agent.motor_ixs),
        's_h': number_of_connections(cm,agent.sensor_ixs, agent.hidden_ixs),
        'h_h': number_of_connections(cm,agent.hidden_ixs, agent.hidden_ixs),
        'h_m': number_of_connections(cm,agent.hidden_ixs, agent.motor_ixs)
        }
    return pd.DataFrame(num_connections, index = [1])

def LSCC(G):
    # largest strongly connected component using networkx graph
    LSCC = max(nx.strongly_connected_components(G), key=len)
    if len(LSCC) < 2:
        return None
    else:
        return LSCC

def len_LSCC(G):
    LSCC = max(nx.strongly_connected_components(G), key=len)
    return len(LSCC)

def len_LWCC(G):
    LWCC = max(nx.weakly_connected_components(G), key=len)
    return len(LWCC)

def fullStructuralAnalysis(agent, connected_only = True):
    #vvv Todo: connections only for connected node
    df = number_of_connected_nodes_by_type(agent)
    df = df.join(number_of_connections_by_type(agent))
    # Components
    if connected_only == True:
        connected_nodes = list(densely_connected_nodes(agent.cm)) + agent.motor_ixs + agent.sensor_ixs
        cm_connected = agent.cm[np.ix_(connected_nodes, connected_nodes)]
        G = nx.from_numpy_matrix(cm_connected, create_using=nx.DiGraph())
    else:
        G = get_graph(agent)

    #vvv Todo: average over connected hidden nodes for flow and betweenness
    components = {
        'len_LSCC': len_LSCC(G),
        'len_LWCC': len_LWCC(G),
        'flow_hierarchy': flow_hierarchy(G), #Flow hierarchy is defined as the fraction of edges not participating in cycles in a directed graph
        'betweenness_centrality': betweenness_centrality(G),
        'degree_centrality': degree_centrality(G)
        }

    df = df.join(pd.DataFrame(components, index = [1]))

    return df