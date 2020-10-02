import numpy as np
import pandas as pd

'''
Data analysis functions. Signal and Block Catching environments.

'''
# In the block catching task the activity output is organized as follows:
# num_experiments X (num gen x num trials x number of step) 
# where the factor in brackets is a pandas database
# trials cycle though all possible initial conditions as follows:
# 
def getBrainActivity(data, n_agents=121, n_trials=128, n_nodes=8, n_sensors=2,n_hidden=4,n_motors=2, world_height = 34):
    '''
    Function for generating an activity matrices for the animats given outputs from mabe
    The issue is that in MABE the sensors are used to update the hidden nodes and motors within one output time step.
    To correctly analyze them in an information theoretical sense, sensors have to be shifted back by one time step compared to the other nodes.
    It is also important not to count transitions between trials as agent transitions, because the agents are reset at every trial
        Inputs:
            data: a pandas object containing the mabe output from activity recording
            n_agents: number of agents recorded
            n_trials: number of trials for each agent
            n_nodes: total number of nodes in the agent brain (sensors+motrs+hidden)
            n_sensors: number of sensors in the agent brain
            n_hidden: number of hidden nodes between the sensors and motors
            n_motors: number of motors in the agent brain
        Outputs:
            brain_activity: a matrix with the timeseries of activity for each trial of every agent. Dimensions(agents)
    '''
    print('Creating activity matrix from MABE output...')
    n_transitions = world_height
    brain_activity = np.zeros((n_agents,n_trials,1+n_transitions,n_nodes))

    for a in list(range(n_agents)):
        for i in list(range(n_trials)):
            for j in list(range(n_transitions+1)):
                ix = a*n_trials*n_transitions + i*n_transitions + j
                if j==0:
                    sensor = np.fromstring(str(data['input_LIST'][ix]), dtype=int, sep=',')[:n_sensors]
                    hidden = np.zeros(n_hidden)
                    motor = np.zeros(n_motors)
                elif j==n_transitions:
                    sensor = np.zeros(n_sensors) #sensors are set to 0 for last transition
                    hidden = np.fromstring(data['hidden_LIST'][ix-1], dtype=int, sep=',')
                    motor = np.fromstring(data['output_LIST'][ix-1], dtype=int, sep=',')
                else:
                    sensor = np.fromstring(str(data['input_LIST'][ix]), dtype=int, sep=',')[:n_sensors]
                    hidden = np.fromstring(data['hidden_LIST'][ix-1], dtype=int, sep=',')
                    motor = np.fromstring(data['output_LIST'][ix-1], dtype=int, sep=',')
                nodes = np.r_[sensor, motor, hidden]
                brain_activity[a,i,j,:] = nodes
    return brain_activity

def get_genome(genomes, run, agent):
    genome = genomes[run]['GENOME_root::_sites'][agent]
    genome = np.squeeze(np.array(np.matrix(genome)))
    return genome