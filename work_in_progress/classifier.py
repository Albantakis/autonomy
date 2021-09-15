import math

import numpy as np
import pandas as pd
from autonomy.utils import *


def s_m_training_data(agent, n_t=2):
    """
	Function for generating the training data. X = n_t sensor time steps, Y = Motor state
		Inputs:
			n_t: how many sensor time steps back in time
		Outputs:
	"""

    n_trials = agent.brain_activity.shape[0]
    n_times = agent.brain_activity.shape[1]
    nsen = agent.n_sensors
    nmot = agent.n_motors

    sensors = np.empty((0, nsen * n_t))
    motors = np.empty((0, nmot))

    for t in agent.brain_activity:
        tr_motors = t[1:, agent.motor_ixs]
        tr_sensors = t[0:-1, agent.sensor_ixs]
        for n in range(1, n_t):
            add_sensors = np.zeros((n_times - 1, nsen))
            add_sensors[n:, :] = t[0 : -(n + 1), agent.sensor_ixs]
            tr_sensors = np.append(tr_sensors, add_sensors, axis=1)

        sensors = np.append(sensors, tr_sensors, axis=0)
        motors = np.append(motors, tr_motors, axis=0)

    return sensors, motors


def s_t_m_training_data(agent, n_t=1):
    """
    Function for generating the training data. X = Sensor state + time step info in binary, Y = Motor state
        Inputs: agent
        Outputs:
    """

    n_trials = agent.brain_activity.shape[0]
    n_times = agent.brain_activity.shape[1]
    nsen = agent.n_sensors
    nmot = agent.n_motors

    n_tn = math.ceil(math.log2(n_times))
    t_step_encoding = [num2state(i, n_tn) for i in range(n_times - 1)]

    sensors = np.empty((0, nsen * n_t + n_tn))
    motors = np.empty((0, nmot))

    for t in agent.brain_activity:
        tr_sensors = np.array(t_step_encoding)
        tr_motors = t[1:, agent.motor_ixs]
        tr_sensors = np.append(tr_sensors, t[0:-1, agent.sensor_ixs], axis=1)
        for n in range(1, n_t):
            add_sensors = np.zeros((n_times - 1, nsen))
            add_sensors[n:, :] = t[0 : -(n + 1), agent.sensor_ixs]
            tr_sensors = np.append(tr_sensors, add_sensors, axis=1)

        sensors = np.append(sensors, tr_sensors, axis=0)
        motors = np.append(motors, tr_motors, axis=0)

    return sensors, motors


def state_pairs_to_tpm(previous_states, next_states, loli=True):

    """Return a TPM from observed state transitions.

    Arguments:
        previous_states (array-like): An array where the first dimension
            indexes states, such that ``previous_states[i]`` precedes
            ``next_states[i]``.
        next_states (array-like): An array where the first dimension indexes
            states, such that ``next_states[i]`` follows
            ``previous_states[i]``.

    Returns:
        pd.DataFrame: A TPM with unique ``previous_states`` as the index and
        unique ``next_states`` as the columns.

    Example:
        >>> sensors = np.array([
        ...     [0, 0],
        ...     [0, 1],
        ...     [1, 0],
        ...     [1, 0],
        ...     [1, 1],
        ...     [1, 1],
        ... ])
        >>> motors = np.array([
        ...     [1, 0],
        ...     [0, 1],
        ...     [0, 0],
        ...     [0, 1],
        ...     [0, 0],
        ...     [1, 1],
        ... ])
        >>> tpm = state_pairs_to_tpm(sensors, motors)
        >>> tpm
        next      (0, 0)  (0, 1)  (1, 0)  (1, 1)
        previous
        (0, 0)       0.0     0.0     1.0     0.0
        (0, 1)       0.0     1.0     0.0     0.0
        (1, 0)       0.5     0.5     0.0     0.0
        (1, 1)       0.5     0.0     0.0     0.5
        >>> # The underlying NumPy array can be accessed with the
        >>> # `.values` attribute:
        >>> tpm.values
        array([[0. , 0. , 1. , 0. ],
               [0. , 1. , 0. , 0. ],
               [0.5, 0.5, 0. , 0. ],
               [0.5, 0. , 0. , 0.5]])
    """
    if loli:
        previous_states = pd.Categorical(
            [state2num(states) for states in previous_states]
        )
        next_states = pd.Categorical([state2num(states) for states in next_states])
    else:
        previous_states = pd.Categorical([tuple(states) for states in previous_states])
        next_states = pd.Categorical([tuple(states) for states in next_states])

    tpm = pd.crosstab(previous_states, next_states, normalize="index")
    tpm.index.name = "previous"
    tpm.columns.name = "next"
    return tpm


def s_t_m_tpm(agent, n_t=2, count_time=False):
    if count_time:
        sensors, motors = s_t_m_training_data(agent, n_t=n_t)
    else:
        sensors, motors = s_m_training_data(agent, n_t=n_t)

    tpm = state_pairs_to_tpm(sensors, motors)
    rank = len(tpm) / ((2 * agent.n_sensors) ** n_t)
    av_accuracy = np.mean(tpm.max(axis=1))

    return tpm, rank, av_accuracy


def temporal_depth(agent, count_time=False):
    accuracy = []
    rank = []
    for ps in range(1, agent.n_timesteps):
        tpm, rank_ps, accuracy_ps = s_t_m_tpm(agent, n_t=ps, count_time=count_time)
        accuracy.append(accuracy_ps)
        rank.append(rank_ps)

    return accuracy, rank
