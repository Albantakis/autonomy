import pyphi
import numpy as np
import networkx as nx
import yaml
from pathlib import Path

from plotting import *
from utils import *
from structuralAgentAnalysis import *

### This is the general agent object based on a TPM and CM and nothing else necessary. However, it should be possible to give other useful parameters

class Agent:
    """
    Represents an agent.

    Args:
        tpm (np.ndarray):
            The agent's 2-D transition probability matrix.
        
    Keyword Args: 
        cm (np.ndarray):
            The agent's connectivity matrix.
        experiment (Experiment): 
            The experiment the agent was evolved to.
        
    Attributes:
        sensor_ixs (list): Indices of Sensors in TPM 
        motor_ixs (list): Indices of Motors in TPM
        hidden_ixs (list): Indices of Motors in TPM
        tpm
        cm
        fitness

    Possible Attribute:

        ...
    """

    def __init__(self, tpm, cm = False, agent_params = False, activity = False):
        '''
        Called by pyAgents.Agent(...)

        Keyword Args:
            agent_params: either string that leads to yml file with parameters or dictionary of parameters.
        ''' 
        if agent_params:
            self._get_agent_params(agent_params)
        
        # TODO: everything hast to come after get_agent_params or it is overwritten. Fix.
        self.tpm = tpm
        self.cm = cm
        
        self.activity = activity
        self.n_nodes = len(tpm[1])

        # TODO: Test that TPM dimsensions and sensor, motor indices, num nodes all fit together

    def _get_agent_params(self, agent_params):
        print(agent_params)
        path = str(Path(__file__).parent) + "/Phenotypes/" + agent_params + '.yml'
        print(path)
        if isinstance(agent_params, str):
            try: 
                version = [int(x) for x in yaml.__version__.split('.')]
                version_float = version[0] + 0.1*version[1]
                if version_float < 5.1: #For Pyanimats environment
                    with open(path, 'rt') as f:
                        params = yaml.load(f)
                else:
                    with open(path, 'rt') as f:
                        params = yaml.full_load(f)
                print(params)
            except FileNotFoundError:
                print("No parameter file to load.")
        else:
            params = agent_params

        self.__dict__ = params
       

    def infer_cm(self):
        pyphi.infer_cm(self.tpm)

    def __len__(self):
        # size of agent in world
        return self.length

    def set_x(self, position):
        # function for setting the current x position of the agent
        self.x = position

    def set_y(self, position):
        # function for setting the current y position of the agent
        self.y = position

    def create_agent(tpm, cm, brain_activity, fitness = 0., task = None, id_label = None, params = {}):

        agent = Agent(params)
        agent.save_brain(tpm, cm)
        agent.save_brain_activity(brain_activity)
        agent.save_task(task)
        agent.save_id(id_label)
        agent.save_fitness(fitness)
        return agent

    def save_task(self, task):
        self.task = task

    def save_id(self, id_label):
        self.id = id_label

    def save_fitness(self, fitness):
        self.fitness = fitness

    def save_brain(self, TPM, cm, node_labels=[]):
        '''
        Function for giving the agent a brain (pyphi network) and a graph object
            Inputs:
                TPM: a transition probability matrix readable for pyPhi
                cm: a connectivity matrix readable for pyPhi
                node_labels: list of labels for nodes (if empty, standard labels are used)
            Outputs:
                no output, just an update of the agent object
        '''
        if not len(node_labels)==self.n_nodes:
            node_labels = []
            # standard labels for up to 10 nodes of each kind
            sensor_labels = ['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10']
            motor_labels = ['M1','M2','M3','M4','M5','M6','M7','M8','M9','M10']
            hidden_labels = ['A','B','C','D','E','F','G','H','I','J']

            # defining labels for each node type
            s = [sensor_labels[i] for i in list(range(self.n_sensors))]
            m = [motor_labels[i] for i in list(range(self.n_motors))]
            h = [hidden_labels[i] for i in list(range(self.n_hidden))]

            # combining the labels
            node_labels.extend(s)
            node_labels.extend(m)
            node_labels.extend(h)

        # defining the network using pyphi
        network = pyphi.Network(TPM, cm, node_labels=node_labels)
        self.brain = network
        self.TPM = TPM
        self.cm = cm
        self.connected_nodes = sum(np.sum(cm,0)*np.sum(cm,1)>0)
        
        # defining a graph object based on the connectivity using networkx
        G = nx.from_numpy_matrix(cm, create_using=nx.DiGraph())
        mapping = {key:x for key,x in zip(range(self.n_nodes),node_labels)}
        G = nx.relabel_nodes(G, mapping)
        self.brain_graph = G
        self.len_LSCC = len_LSCC(G)

        # saving the labels and indices of sensors, motors, and hidden to animats
        self.node_labels = node_labels
        self.sensor_ixs = list(range(self.n_sensors))
        self.sensor_labels = [node_labels[i] for i in self.sensor_ixs]
        self.motor_ixs = list(range(self.n_sensors,self.n_sensors+self.n_motors))
        self.motor_labels = [node_labels[i] for i in self.motor_ixs]
        self.hidden_ixs = list(range(self.n_sensors+self.n_motors,self.n_sensors+self.n_motors+self.n_hidden))
        self.hidden_labels = [node_labels[i] for i in self.hidden_ixs]

    def save_brain_activity(self,brain_activity):
        '''
        Function for saving brain activity to agent object
            Inputs:
                brain activity as 3D array (trials x times x nodes)
            Outputs:
                no output, just an update of the Agent object
        '''
        assert brain_activity.shape[2]==self.n_nodes, "Brain history does not match number of nodes = {}".format(self.n_nodes)
        self.brain_activity = np.array(brain_activity).astype(int)
        self.n_trials = brain_activity.shape[0]
        self.n_timesteps = brain_activity.shape[1]

# -------------------- PLOTTING --------------------------------------

    def plot_brain(self, state=None, ax=None):
        '''
        Function for plotting the brain of an animat.
        ### THIS FUNCTION ONLY WORKS WELL FOR ANIMATS WITH 7 or 8 NODES (2+2+4) ###
            Inputs:
                state: the state of the animat for plotting (alters colors to indicate activity)
                ax: for specifying which axes the graph should be plotted on

            Outputs:
                no output, just calls plotting.py function for plotting
        '''
        plot_animat_brain(self.brain.cm, self.brain_graph, state, ax)