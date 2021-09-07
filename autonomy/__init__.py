# __init__.py

"""
========
autonomy
========

Autonomy is a Python toolbox for computing measure of autonomy on
simple artificial agents.

If you use this software in your research, please cite the paper::

    Albantakis (2021) Quantifying the autonomy of structurally diverse automata:
    a comparison of candidate measures. Forthcoming.

To report issues, please send an email to albantakis@wisc.edu.

Usage
~~~~~
The |Agent| object is the main object on which computations are performed. It
represents the causal model of the agent as a transition probability matrix.
"""

import pyphi

pyphi.config.CLEAR_SUBSYSTEM_CACHES_AFTER_COMPUTING_SIA = True
pyphi.config.PROGRESS_BARS = False

from . import (
    ShapleyValues,
    causalAgentAnalysis,
    dynamicalAgentAnalysis,
    informationAgentAnalysis,
    plotting,
    pyAgent,
    structuralAgentAnalysis,
    utils,
)
from .__about__ import *  # pylint: disable=wildcard-import
from .plotting import plot_animat_brain
from .pyAgent import LOD, Agent

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
