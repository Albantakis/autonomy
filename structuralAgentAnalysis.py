
import numpy as np
import pyphi
import networkx as nx
import copy


#####################################################################################################################
### Collection of functions to assess the structural properties of an agent based on its connectivity matrix (cm) ###

# Note: Expects a certain order of nodes: sensors - motors - hidden units (current order of MABE output)

# Todo: 

# - Make function that evaluates all measures and outputs pd.dataframe row (from agent object as input)
# - spectral density (Banerjee and Jost, 2009)?
# - density of connections and weight distributions (the latter for ANNs)

#####################################################################################################################

def number_of_connected_nodes(cm):
    return np.sum(np.sum(cm,0)*np.sum(cm,1)>0)

def number_of_connected_sensors(cm,n_sensors):
    return np.sum(np.sum(cm[:n_sensors,:],1)>0)

def number_of_connected_motors(cm,n_sensors,n_motors):
    return np.sum(np.sum(cm[:,n_sensors:n_sensors+n_motors],0)>0)

def number_of_densely_connected_nodes(cm_agent,allow_self_loops=False):
    cm = copy.copy(cm_agent)
    if not allow_self_loops:
        for i in range(len(cm)):
            cm[i,i] = 0
    return np.sum((np.sum(cm,0)*np.sum(cm,1))>0)

def densely_connected_nodes(cm_agent,allow_self_loops=False):
    # Hidden nodes with inputs and outputs
    cm = copy.copy(cm_agent)
    if not allow_self_loops:
        for i in range(len(cm)):
            cm[i,i] = 0
    return np.where((np.sum(cm,0)*np.sum(cm,1))>0)[0]

def number_of_sensor_hidden_connections(cm,n_sensors,n_motors):
    return np.sum(cm[:n_sensors,n_sensors+n_motors:]>0)

def number_of_sensor_motor_connections(cm,n_sensors,n_motors):
    return np.sum(cm[:n_sensors,n_sensors:n_sensors+n_motors]>0)

def number_of_hidden_hidden_connections(cm,n_sensors,n_motors):
    return np.sum(cm[n_sensors+n_motors:,n_sensors+n_motors:]>0)

def number_of_hidden_motor_connections(cm,n_sensors,n_motors):
    return np.sum(cm[n_sensors+n_motors:,n_sensors:n_sensors+n_motors]>0)

# largest strongly connected component using networkx graph
def LSCC(G):
    LSCC = max(nx.strongly_connected_components(G), key=len)
    if len(LSCC) < 2:
        return None
    else:
        return LSCC

def len_LSCC(G):
    LSCC = max(nx.strongly_connected_components(G), key=len)
    return len(LSCC)