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

def I_PRED(agent, n_t = 1):
	H_A = set_entropy(agent)
	H_AA = joint_entropy(agent, n_t = n_t)
	return 2*H_A - H_AA


### Autonomy measures

def A_m_sensors(agent, n_t = 5):
	# H(HM_t|S_t-1,...S_t-n) - H(HM_t|HM_t-1, S_t-1,...S_t-n)
	# Make node_in_arrays
	# HM_t, S_t-1, ..., S_t-n
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
	
	return entropy(prob_j1) - entropy(prob_c1) - entropy(prob_j2) + entropy(prob_c2)



	
### Respresentation (requires environment states)