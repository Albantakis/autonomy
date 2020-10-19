import numpy as np
from scipy.stats import entropy
from pyphi import convert
from utils import *

#####################################################################################################################
### Collection of functions to assess the information theoretical properties of an agent based on its activity 	  ###

# Note: Activity is a 3D array (trials x times x nodes)

# Todo: 
# - Autonomy A_m (Bertschiger 2008)
# - causal closure (NTIC)
# - Respresentation R (see Hintze et al., 2018 for latest application)
# - Make function that evaluates all measures and outputs pd.dataframe row
# - Fix computation by using correct distributions. Compute MI with mutual 

#####################################################################################################################

### Entropies

def set_entropy(agent, node_indices = None, exclude_first_last = True):
	# evaluates nodes within one ts
	prob_dist = get_st_prob_distribution(agent, exclude_first_last = exclude_first_last, node_indices = node_indices)
	return entropy(prob_dist)

def joint_entropy(agent, node_ind_pair = None, n_t = 1):
	# evaluates sets of nodes across n_t timesteps
	prob_dist = get_trans_prob_distribution(agent, node_ind_pair = node_ind_pair, n_t = n_t)
	return entropy(prob_dist)


def I_MULTI(agent):
	# sum_i(H(V_i)) - H(V)
	H_Vi = [set_entropy(agent, node_indices = n) for n in range(agent.n_nodes)]
	H_V = set_entropy(agent)
	return sum(H_Vi) - H_V 

### Mutual information measures
def I_SMMI(agent, n_t = 1):
	H_S = set_entropy(agent, node_indices = agent.sensor_ixs)
	H_M = set_entropy(agent, node_indices = agent.motor_ixs)
	H_SM = joint_entropy(agent, node_ind_pair = [agent.sensor_ixs, agent.motor_ixs], n_t = n_t)
	return H_S + H_M - H_SM


def I_PRED(agent, n_t = 1, woSen = False):
	# A* in Bertschiger et al., 2008 (Autonomy measure if the system does not depend on the environment)
	if woSen:
		internal_nodes = agent.hidden_ixs + agent.motor_ixs
		H_A = set_entropy(agent, node_indices = internal_nodes)
		H_AA = joint_entropy(agent, node_ind_pair = [internal_nodes, internal_nodes], n_t = n_t)
	else:
		H_A = set_entropy(agent)
		H_AA = joint_entropy(agent, n_t = n_t)
	return 2*H_A - H_AA

def IC(agent):
	# Information closure based on sensors
	internal_nodes = agent.hidden_ixs + agent.motor_ixs
	H_HH = joint_entropy(agent, node_ind_pair = [internal_nodes, internal_nodes], n_t = 1)
	H_H = set_entropy(agent, node_indices = internal_nodes)
	H_HHS = joint_entropy(agent, node_ind_pair = [internal_nodes, internal_nodes + agent.sensor_ixs], n_t = 1)
	H_HS = set_entropy(agent, node_indices = internal_nodes + agent.sensor_ixs)

	return (H_HH - H_H) - (H_HHS - H_HS)

def NTIC_m(agent, n_t = 5):
	# Nontrivial information closure considering m time steps of the environment (here sensors)
	# A^* = A_m + NTIC_m  --> NTIC_m = A^* - A_m with A^* = I_PRED
	# Supposedly the degree with which the system models the environment (but can be positive for FF systems)
	return I_PRED(agent, n_t = 1, woSen = True) - A_m_sensors(agent, n_t = n_t)


### Autonomy measures

def A_m_sensors(agent, n_t = 5):	
	# H(HM_t|S_t-1,...S_t-n) - H(HM_t|HM_t-1, S_t-1,...S_t-n)
	# Note that for deterministic systems the second part is 0
	# Make node_in_arrays
	# HM_t, S_t-1, ..., S_t-n
	# n_t = m (number of past times steps of the environment taken into account)
	node_ind_array_j1 = [agent.hidden_ixs + agent.motor_ixs]
	for n in range(n_t):
		node_ind_array_j1.append(agent.sensor_ixs)
 	
	states, count = get_unique_multiple_ts(agent.brain_activity, return_counts = True, node_ind_array = node_ind_array_j1, n_t = n_t)
	prob_j1 = count/sum(count)

 	#HS_t-1, ..., S_t-n
	node_ind_array_c1 = []
	for n in range(n_t):
		node_ind_array_c1.append(agent.sensor_ixs)

	states, count = get_unique_multiple_ts(agent.brain_activity, return_counts = True, node_ind_array = node_ind_array_c1, n_t = n_t-1)
	prob_c1 = count/sum(count)

	# HM_t, HMS_t-1, ..., S_t-n
	node_ind_array_j2 = [agent.hidden_ixs + agent.motor_ixs]
	node_ind_array_j2.append(agent.hidden_ixs + agent.motor_ixs + agent.sensor_ixs)
	for n in range(1,n_t):
		node_ind_array_j2.append(agent.sensor_ixs)

	states, count = get_unique_multiple_ts(agent.brain_activity, return_counts = True, node_ind_array = node_ind_array_j2, n_t = n_t)
	prob_j2 = count/sum(count)

	# HMS_t-1, ..., S_t-n	
	node_ind_array_c2 = [agent.hidden_ixs + agent.motor_ixs + agent.sensor_ixs]
	for n in range(1,n_t):
		node_ind_array_c2.append(agent.sensor_ixs)
	
	states, count = get_unique_multiple_ts(agent.brain_activity, return_counts = True, node_ind_array = node_ind_array_c2, n_t = n_t-1)
	prob_c2 = count/sum(count)
	
	print(entropy(prob_j2) - entropy(prob_c2)) #This should be zero in deterministic system, but is not because different distributions are used to compute the entropies.

	return entropy(prob_j1) - entropy(prob_c1) - entropy(prob_j2) + entropy(prob_c2)




	
### Respresentation (requires environment states)