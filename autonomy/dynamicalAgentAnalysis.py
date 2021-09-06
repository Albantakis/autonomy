import numpy as np
import numpy.random as ran
import pyphi

from .utils import *
from .causalAgentAnalysis import fix_TPM_dim

#######################################################################################################################
### Collection of functions to assess the dynamical properties of an agent 											###
#######################################################################################################################

# specific utils
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def state_evolution(current_state, tpm):
	# inputs: current_state in int, tpm state-by-node 2D array
	state_evolution = []
	while (current_state not in state_evolution) and len(state_evolution) < len(tpm):
		state_evolution.append(current_state)
		current_state = state2num(tpm[current_state])

	return state_evolution

def all_possible_transients(agent, numerical = True):
	# set agent in all possible states (TPM inputs)
	# keep sensors fixed throughout the evolution
	
	tpm, _, _ = fix_TPM_dim(agent, motors = True)

	ind_hm = np.sort(agent.hidden_ixs + agent.motor_ixs)
	
	num_input_states = 2**agent.n_sensors

	transients = []

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
			st_evolution = state_evolution(s, tpm_cond)

			if numerical is False:
				st_evolution = [num2state(r, len(ind_hm)) for r in st_evolution]

			transients.append(st_evolution)

	return transients


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

# measures
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def number_of_unique_transitions(agent):
	# Sensor + Hidden -> Hidden + Motor (Motors do not feed back and sensors are set by the environment)
	ind_hs = tuple(agent.hidden_ixs + agent.sensor_ixs)
	ind_hm = tuple(agent.hidden_ixs + agent.motor_ixs)
	node_ind_pair = [ind_hs, ind_hm]

	transitions = get_unique_transitions(agent, return_counts = False, node_ind_pair = node_ind_pair, n_t = 1)
		
	return len(transitions)


def LZ_complexity(agent, dim='space', data = None, threshold=0, shuffles=10):
	# get 2D activity data
	if data is None:
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

def LZ_transient_data(agent):
	
	transients = all_possible_transients(agent, numerical = False)
	data = []
	for t in transients:
		if len(t) > 1:
			for st in t[1:]:
				data.append(st)

	return np.array(data)


def evaluate_transient_length(agent):
	transients = all_possible_transients(agent)

	transient_length = [len(t) for t in transients]

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
		'nLZ_tr_time': LZ_complexity(agent, dim = 'time', data = LZ_transient_data(agent)),
		'nLZ_tr_space': LZ_complexity(agent, dim = 'space', data = LZ_transient_data(agent)),
		}

	df = pd.DataFrame(dynamical_measures, index = [1])

	if save_agent:
		agent.dynamical_analysis = df

	return df


def emptyDynamicalAnalysis(index_num = 1):
    df = pd.DataFrame(dtype=float, columns = ['#Transitions','maxTL', 'avTL', 'nLZ_time', 'nLZ_space', 'nLZ_tr_time', 'nLZ_tr_space'], index = range(index_num))
    return df