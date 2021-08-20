import numpy as np
import numpy.random as ran
import pyphi
from utils import *
from causalAgentAnalysis import fix_TPM_dim

#######################################################################################################################
### Collection of functions to assess the dynamical properties of an agent 											###
#######################################################################################################################

def number_of_unique_transitions(agent):
	# Sensor + Hidden -> Hidden + Motor (Motors do not feed back and sensors are set by the environment)
	ind_hs = tuple(agent.hidden_ixs + agent.sensor_ixs)
	ind_hm = tuple(agent.hidden_ixs + agent.motor_ixs)
	node_ind_pair = [ind_hs, ind_hm]

	transitions = get_unique_transitions(agent, return_counts = False, node_ind_pair = node_ind_pair, n_t = 1)
		
	return len(transitions)


def LZ_algorithm(string):
	d={}
	w = ''
	i=1
	for c in string:
		wc = w + c
		if wc in d:
			w = wc
		else:
			d[wc]=wc
			w = c
		i+=1
	return len(d)


def LZ_complexity(agent,dim='space',threshold=0,shuffles=10):
	# get 2D activity data
	data = np.array(activity_list_concurrent_inputs(agent))

	#sort columns by STD
	sumOnes = np.sum(data, axis = 0)
	Irank= sumOnes.argsort()[::-1]
	data = data[:,Irank]

	# setting up variables for concatinating the data over time or space dimension
	if dim=='space':
		data = np.transpose(data)

	d1, d2 = data.shape

	# making the concatenated (1D) string for calculating complexity
	s = ''
	for j in range(d2):
		for i in range(d1):
			if data[i,j]>threshold:
				s+='1'
			else:
				s+='0'

	# calculating raw LZ
	lz_raw = LZ_algorithm(s)

	# normalization by random sequence
	randoms = []
	new_s = list(s)
	for i in range(shuffles):
		ran.shuffle(new_s)
		randoms.append(LZ_algorithm(new_s))

	return lz_raw/np.max(randoms)


def evaluate_transient_length(agent):
	# set agent in all possible states (TPM inputs)
	# keep sensors fixed throughout the evolution
	tpm, cm, ind = fix_TPM_dim(agent, motors = True)

	ind_hm = np.sort(agent.hidden_ixs + agent.motor_ixs)

	num_input_states = 2**agent.n_sensors

	transient_length = []

	for si in range(num_input_states):
			sensor_state = num2state(si, agent.n_sensors)
			
			full_state = np.array([0 for i in range(agent.n_nodes)])
			full_state[agent.sensor_ixs] = sensor_state

			tpm_cond = pyphi.tpm.condition_tpm(pyphi.convert.to_multidimensional(tpm), agent.sensor_ixs, full_state)
			tpm_cond = pyphi.convert.to_2dimensional(tpm_cond)
			tpm_cond = np.array(tpm_cond)[:,ind_hm]

			# start from all initial conditions, evolve until states repeat
			for s in range(len(tpm_cond)):
				# initialize state_evolution with initial state
				state_evolution = []
				current_state = s
				while (current_state not in state_evolution) and len(state_evolution) < len(tpm_cond):
					state_evolution.append(current_state)
					current_state = state2num(tpm_cond[current_state])
				
				transient_length.append(len(state_evolution))

	max_transient_length = np.max(transient_length)
	av_transient_length = np.mean(transient_length)
	
	return max_transient_length, av_transient_length


# All
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def fullDynamicalAnalysis(agent, save_agent = False):
	
	maxTl, avTl = evaluate_transient_length(agent)

	dynamical_measures = {
    	'#Transitions': number_of_unique_transitions(agent),
		'maxTL': maxTl,
		'avTL': avTl,
		'nLZ_time': LZ_complexity(agent, dim = 'time'),
		'nLZ_space': LZ_complexity(agent, dim = 'space'),
		}

	df = pd.DataFrame(dynamical_measures, index = [1])

	if save_agent:
		agent.dynamical_analysis = df

	return df


def emptyDynamicalAnalysis(index_num = 1):
    df = pd.DataFrame(dtype=float, columns = ['#Transitions','maxTL', 'avTL', 'nLZ_time', 'nLZ_space'], index = range(index_num))
    return df