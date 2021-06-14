import pyphi
import numpy as np
import yaml
from pathlib import Path

from plotting import *
from utils import *
from structuralAgentAnalysis import *

### This is the general agent object based on a TPM and (inferred) CM and nothing else necessary. 

class LOD:
    "Line of descent paramenters, e.g.: fitness, generation, run"
    def __init__(self, LOD_dict):
        self.__dict__ = LOD_dict

class Agent:
    """
    Represents an agent.

    Args:
        tpm (np.ndarray):
            The agent's 2-D transition probability matrix.
        
    Keyword Args: 
        cm (np.ndarray):
            The agent's connectivity matrix.
        activity (np.array): 
            vvv
        
    Attributes:
        sensor_ixs (list): Indices of Sensors in TPM 
        motor_ixs (list): Indices of Motors in TPM
        hidden_ixs (list): Indices of Motors in TPM
        tpm
        cm

    Possible Attribute:
        LOD
    """

    def __init__(self, tpm, cm = None, activity = None, params = None, add_attr = None, LOD_dict = None):
        '''
        Called by pyAgents.Agent(...)

        Keyword Args:
            agent_params: either string that leads to yml file with parameters or dictionary of parameters.
        ''' 
        if params:
            self._get_agent_params(params)

        if add_attr:
            self._get_agent_params(add_attr)

        if LOD: 
            self.LOD = LOD(LOD_dict)
        
        self.tpm = tpm

        if cm is None: 
            cm = self._infer_cm()
        self.cm = cm
        
        self.activity = activity
        self.n_nodes = len(tpm[1])

        # TODO: Test that TPM dimsensions and sensor, motor indices, num nodes all fit together

    def _get_agent_params(self, agent_params):
        print(agent_params)
        if isinstance(agent_params, str):
            path = str(Path(__file__).parent) + "/Phenotypes/" + agent_params + '.yml'
            print(path)
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

        self.__dict__.update(params)
       

    def _infer_cm(self):
        return pyphi.tpm.infer_cm(pyphi.convert.to_multidimensional(self.tpm))

    def LSCC(self):
        G = get_graph(self)
        return LSCC(G)
    
# -------------------- PLOTTING --------------------------------------

    def plot_brain(self, state=None, ax=None):
        '''
        Function for plotting the brain of an animat.
            Inputs:
                state: the state of the animat for plotting (alters colors to indicate activity)
                ax: for specifying which axes the graph should be plotted on

            Outputs:
                no output, just calls plotting.py function for plotting
        '''
        plot_animat_brain(self.brain.cm, state, ax)