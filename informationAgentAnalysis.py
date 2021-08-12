import numpy as np
from scipy.stats import entropy
from pyphi import convert
from utils import *
from itertools import combinations

#####################################################################################################################
### Collection of functions to assess the information theoretical properties of an agent based on its activity 	  ###

# Todo: 
# - Use TSE Complexity from dit?

#####################################################################################################################

# Entropies
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def set_entropy(agent, node_indices = None):
	# evaluates nodes within one ts
	prob_dist = get_st_prob_distribution(agent, node_indices = node_indices)
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


# Mutual information measures
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def I_SMMI(agent, n_t = 1):
	H_S = set_entropy(agent, node_indices = agent.sensor_ixs)
	H_M = set_entropy(agent, node_indices = agent.motor_ixs)
	# Note: sampling is off by n_t states for the joint entropy with n_t > 0
	H_SM = joint_entropy(agent, node_ind_pair = [agent.sensor_ixs, agent.motor_ixs], n_t = n_t)
	return H_S + H_M - H_SM


def I_PRED(agent, n_t = 1, noSen = False):
	# A* in Bertschiger et al., 2008 (Autonomy measure if the system does not depend on the environment)
	if noSen:
		internal_nodes = agent.hidden_ixs + agent.motor_ixs
		H_A = set_entropy(agent, node_indices = internal_nodes)
		# Note: sampling is off by n_t states for the joint entropy with n_t > 0
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
	return I_PRED(agent, n_t = 1, noSen = True) - A_m_sensors(agent, n_t = n_t)


# Autonomy measures
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def A_m_sensors(agent, n_t = 5):	
	# H(OM_t|S_t-1,...S_t-n) - H(OM_t|OM_t-1, S_t-1,...S_t-n)
	# Note that for deterministic systems the second part is 0
	# Make node_in_arrays
	# OM_t, S_t-1, ..., S_t-n
	# n_t = m (number of past times steps of the environment taken into account)
	# if n_t = 1, A_m_sensors, becomes predictive information

	# OM_t, S_t-1, ... , S_t-n
	node_ind_array_joint_1 = [agent.hidden_ixs + agent.motor_ixs]
	for n in range(n_t):
		node_ind_array_joint_1.append(agent.sensor_ixs)
	
	states, count = get_unique_multiple_ts(agent, return_counts = True, node_ind_array = node_ind_array_joint_1, n_t = n_t)
	prob_joint_1 = count/sum(count)

 	#S_t-1, ..., S_t-n
	node_ind_array_cond_1 = []
	for n in range(n_t):
		node_ind_array_cond_1.append(agent.sensor_ixs)

	states, count = get_unique_multiple_ts(agent, return_counts = True, node_ind_array = node_ind_array_cond_1, n_t = n_t-1)
	prob_cond_1 = count/sum(count)

	# OM_t, OMS_t-1, S_t-2, ..., S_t-n
	node_ind_array_joint_2 = [agent.hidden_ixs + agent.motor_ixs]
	node_ind_array_joint_2.append(agent.hidden_ixs + agent.motor_ixs + agent.sensor_ixs)
	for n in range(1,n_t):
		node_ind_array_joint_2.append(agent.sensor_ixs)

	states, count = get_unique_multiple_ts(agent, return_counts = True, node_ind_array = node_ind_array_joint_2, n_t = n_t)
	prob_joint_2 = count/sum(count)

	# OMS_t-1, ..., S_t-n	
	node_ind_array_cond_2 = [agent.hidden_ixs + agent.motor_ixs + agent.sensor_ixs]
	for n in range(1,n_t):
		node_ind_array_cond_2.append(agent.sensor_ixs)
	
	states, count = get_unique_multiple_ts(agent, return_counts = True, node_ind_array = node_ind_array_cond_2, n_t = n_t-1)
	prob_cond_2 = count/sum(count)
	
	#print(entropy(prob_joint_2) - entropy(prob_cond_2)) #This should be zero in deterministic system, but is not because different distributions are used to compute the entropies.

	return entropy(prob_joint_1) - entropy(prob_cond_1) - entropy(prob_joint_2) + entropy(prob_cond_2)

# TSE Complexity
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def sub_entropies(agent, k):
    """
    Compute the mean entropy of all subsets of size k of the agents hidden nodes.
    """

    subsets = [s for s in combinations(agent.hidden_ixs, k)]
    print(subsets)
    subsetH = sum([set_entropy(agent, node_indices = s) for s in subsets])

    subsetH = subsetH / len(subsets)
    
    return subsetH

def TSE_Complexity(agent):
	N = agent.n_hidden
	H_N = set_entropy(agent, node_indices = agent.hidden_ixs)  
	TSE = sum([sub_entropies(agent, k) - k/N * H_N for k in range(1, N)])
	return TSE


def fullInformationAnalysis(agent, save_agent = False):
    
    information_measures = {
        'H': set_entropy(agent),
        'I_SMMI': I_SMMI(agent, n_t = 1),
        'I_PRED': I_PRED(agent, n_t = 1),
        'I_PRED_OM': I_PRED(agent, n_t = 1, noSen = True),
        'A_4': A_m_sensors(agent, n_t = 4),
        'IC': IC(agent),
        'NTIC_4': NTIC_m(agent, n_t = 4),
        'MI': I_MULTI(agent),
        'TSE': TSE_Complexity(agent)
        }

    df = pd.DataFrame(information_measures, index = [1])

    if save_agent:
    	agent.info_analysis = df

    return df

def emptyInformationAnalysis(index_num = 1):
    df = pd.DataFrame(dtype=float, columns = ['H','I_SMMI', 'I_PRED', 'I_PRED_OM', 'A_4', 'IC', 'NTIC_4', 'MI', 'TSE'], index = range(index_num))
    return df