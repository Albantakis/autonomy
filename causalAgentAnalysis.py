import numpy as np
import pyphi

#######################################################################################################################
### Collection of functions to assess the causal properties of an agent based on its transition probability matrix  ###

# Todo: 
# - A^* (like max ent, but perturbed by marginal distribution?)
# - multi information (causal)
# - Phi
# - sum of small phi
# - Actual causation + causal history?
# - Make function that evaluates all measures and outputs pd.dataframe row

#######################################################################################################################

def sys_entropy(network):
	sbs_tpm = convert.state_by_node2state_by_state(network.tpm)
	avg_repertoire = np.mean(sbs_tpm, 0)

	return entropy(avg_repertoire, base = 2.)

#Todo: fix
# def node_entropy(network):
# 	avg_prob = np.mean(network.tpm.reshape([number_of_states]+[number_of_nodes], order = 'F'),0)

# 	return np.mean([entropy([p, 1-p], base = 2.) for p in avg_prob])



def effective_information(network):
	return pyphi.macro.effective_info(network)

def sum_of_small_phi():

def main_complex():

def average_Phi():