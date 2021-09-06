import json
from collections import UserDict
from copy import deepcopy
from pathlib import Path

import pyphi
import yaml

from .plotting import *
from .structuralAgentAnalysis import LSCC
from .utils import *

### This is the general agent object based on a TPM and (inferred) CM and nothing else necessary. 

class LOD(UserDict):
    "Line of descent paramenters, e.g.: fitness, generation, run"

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

        
    def to_dict(self):
        """Return a dictionary-only representation of the Agent."""
        # Make a copy to prevent in-place modification
        dct = deepcopy(self.__dict__)

        def convert(value):
            if isinstance(value, pd.DataFrame):
                return value.to_dict()
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, LOD):
                return dict(value)
            return value

        return {
            key: convert(value) for key, value in dct.items()
        }
        

    @classmethod
    def from_dict(cls, dct):
        """Create an Agent from a dictionary representation."""
        # Make a copy to prevent in-place modification
        dct = deepcopy(dct)

        dct['LOD_dict'] = dct.pop('LOD')
        kwargs = {
            kwarg: dct.pop(kwarg)
            for kwarg in ['tpm', 'cm', 'activity', 'LOD_dict']
        }

        agent = cls(**kwargs)
        agent.__dict__.update(dct)
        return agent

        
    def write(self, path):
        """Write this Agent to disk as a JSON file.

        Args:
            path (path_like): The filepath to write to. Suffix will be changed to '.json'.
        
        Returns:
            path (Path): The path that was written to.
        """
        path = Path(path).with_suffix('.json')
        with open(path, mode='wt') as f:
            json.dump(self.to_dict(), f)
        return path
    

    @classmethod
    def read(cls, path):
        """Return an Agent stored in a JSON file on disk.

        Args:
            path (path_like): The filepath to read from.

        Returns:
            Agent: The stored agent.
        """
        with open(path, mode='rt') as f:
            return cls.from_dict(json.load(f))


    
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
        plot_animat_brain(self, state=state, ax=ax)
