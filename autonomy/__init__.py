"""
========
autonomy
========
Autonomy is a Python toolbox for computing measure of autonomy on
simple artificial agents.
If you use this software in your research, please cite the paper:
    Albantakis (2021) Quantifying the autonomy of structurally diverse 
    automata: acomparison of candidate measures. Forthcoming.
To report issues, please send an email to albantakis@wisc.edu.

Usage
~~~~~
The |Agent| object is the main object on which computations are performed. It
represents the causal model of the agent as a transition probability matrix.
"""

import os
import pyphi

pyphi.config.CLEAR_SUBSYSTEM_CACHES_AFTER_COMPUTING_SIA = True
pyphi.config.PROGRESS_BARS = False

from .__about__ import *  # pylint: disable=wildcard-import

from . import (
    pyAgent,
    causalAgentAnalysis,
    dynamicalAgentAnalysis,
    informationAgentAnalysis,
    structuralAgentAnalysis,
    plotting,
    ShapleyValues,
    utils,
)
from .plotting import plot_animat_brain

__all__ = [
    "Agent",
    "LOD",
]

print(
        """
Welcome to autonomy!
To report issues, please send an email to albantakis@wisc.edu.
"""
    )