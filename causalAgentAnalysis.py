import numpy as np
import pyphi
from utils import *

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
	return

def main_complex():
	return

def average_Phi_3_0(agent, save_agent = False):
    # PathFollow agents TPMs are only specified for motors = 000 to save memory, motors do not feed back
    if len(agent.tpm) is not (2**agent.n_nodes):
        ind = np.sort(agent.sensor_ixs+agent.hidden_ixs)
        tpm = np.array(agent.tpm)[:,ind]
        cm = agent.cm[:,ind][ind]
    else:
        tpm = agent.tpm
        cm = agent.cm
        ind = None

    states,counts = get_unique_states_PF(agent, node_indices = ind, return_counts = True)
    probs = counts/sum(counts)

    if save_agent:
    	agent.Phi_3_0 = {
    		"states": {},
    		"average_Phi": 0.0
    	}

    network = pyphi.Network(tpm, cm = cm)

    Phi = []
    for s, state in enumerate(states):
        sia = pyphi.compute.major_complex(network, state)
        Phi.append(sia.phi)

        if save_agent:
        	agent.Phi_3_0["states"][s] = {
        		"major_complex": sia.subsystem.nodes,
        		"state": state,
        		"Phi": sia.phi,
        		"num_distinctions": len(sia.ces),
        		"sum_phi": sum([d.phi for d in sia.ces])
        	}

    average_Phi = sum(Phi*probs)
    if save_agent:
     	agent.Phi_3_0["average_Phi"] = average_Phi
    
    return average_Phi