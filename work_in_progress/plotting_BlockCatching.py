import copy

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

# from pyBlockEnvironment import *
from structural_agent_analysis import densely_connected_nodes
from utils import *


def plot_animat_brain_BC(cm, graph=None, state=None, ax=None):
    """
    Function description
        Inputs:
            inputs:
        Outputs:
            outputs:
    """

    n_nodes = cm.shape[0]
    if n_nodes == 7:
        labels = ["S1", "M1", "M2", "A", "B", "C", "D"]
        pos = {
            "S1": (5, 40),  #'S2': (20, 40),
            "A": (0, 30),
            "B": (20, 30),
            "C": (0, 20),
            "D": (20, 20),
            "M1": (5, 10),
            "M2": (15, 10),
        }
        nodetype = (0, 1, 1, 2, 2, 2, 2)

        ini_hidden = 3

    elif n_nodes == 8:
        labels = ["S1", "S2", "M1", "M2", "A", "B", "C", "D"]
        pos = {
            "S1": (5, 40),
            "S2": (15, 40),
            "A": (0, 30),
            "B": (20, 30),
            "C": (0, 20),
            "D": (20, 20),
            "M1": (5, 10),
            "M2": (15, 10),
        }
        nodetype = (0, 0, 1, 1, 2, 2, 2, 2)
        ini_hidden = 4

    if graph is None:
        graph = nx.from_numpy_matrix(cm, create_using=nx.DiGraph())
        mapping = {key: x for key, x in zip(range(n_nodes), labels)}
        graph = nx.relabel_nodes(graph, mapping)

    state = [1] * n_nodes if state == None else state

    blue, red, green, grey, white = (
        "#6badf9",
        "#f77b6c",
        "#8abf69",
        "#adadad",
        "#ffffff",
    )
    blue_off, red_off, green_off, grey_off = "#e8f0ff", "#ffe9e8", "#e8f2e3", "#f2f2f2"

    colors = np.array(
        [
            [red_off, blue_off, green_off, grey_off, white],
            [red, blue, green, grey, white],
        ]
    )

    node_colors = [colors[state[i], nodetype[i]] for i in range(n_nodes)]
    # Grey Uneffective or unaffected nodes
    cm_temp = copy.copy(cm)
    cm_temp[range(n_nodes), range(n_nodes)] = 0
    unaffected = np.where(np.sum(cm_temp, axis=0) == 0)[0]
    uneffective = np.where(np.sum(cm_temp, axis=1) == 0)[0]
    noeffect = list(set(unaffected).union(set(uneffective)))
    noeffect = [ix for ix in noeffect if ix in range(ini_hidden, ini_hidden + 4)]
    node_colors = [
        node_colors[i] if i not in noeffect else colors[state[i], 3]
        for i in range(n_nodes)
    ]

    #   White isolate nodes
    isolates = [x for x in nx.isolates(graph)]
    node_colors = [
        node_colors[i] if labels[i] not in isolates else colors[0, 4]
        for i in range(n_nodes)
    ]

    self_nodes = [labels[i] for i in range(n_nodes) if cm[i, i] == 1]
    linewidths = [2.5 if labels[i] in self_nodes else 1 for i in range(n_nodes)]

    #     fig, ax = plt.subplots(1,1, figsize=(4,6))
    nx.draw(
        graph,
        with_labels=True,
        node_size=800,
        node_color=node_colors,
        edgecolors="#000000",
        linewidths=linewidths,
        pos=pos,
        ax=ax,
    )


def plot_timestep(agent, world, trial, t, block_patterns=None):

    """
    Function description
        Inputs:
            inputs:
        Outputs:
            outputs:
    """

    # PLOT SCREEN
    fullgame_history = world.get_fullgame_history(block_patterns=block_patterns)

    _, block = world._get_initial_condition(trial)

    wins = world.wins

    n_cols = 3
    n_rows = 18
    col = 0

    plt.subplot2grid((n_rows, n_cols), (0, col), rowspan=18, colspan=1)
    col += 1

    plt.imshow(fullgame_history[trial, t, :, :], cmap=plt.cm.binary)
    plt.xlabel("x")
    plt.ylabel("y")
    win = "WON" if wins[trial] else "LOST"
    direction = "━▶" if block.direction == "right" else "◀━"
    plt.title(
        "Game - Trial: {}, {}, {}, {}".format(block.size, block.type, direction, win)
    )

    state = get_state_tuple(agent, trial, t)
    plt.subplot2grid((n_rows, n_cols), (0, col), rowspan=17, colspan=2)
    agent.plot_brain(state)

    plt.subplot2grid((n_rows, n_cols), (17, col), rowspan=1, colspan=2)
    plt.imshow(np.array(state)[np.newaxis, :], cmap=plt.cm.binary)
    plt.yticks([])
    plt.xticks(range(agent.n_nodes), agent.node_labels)


def plot_trial_cumulative(agent, world, trial):

    world.agent = copy.copy(agent)

    Wagent, block = world._get_initial_condition(trial)
    total_time = world.height  # 35 time steps, 34 updates
    motor_activity = agent.get_motor_activity(trial)

    compressed_block_screen = []
    compressed_agent_screen = []

    ScreenAgent = Screen(world.width, world.height)
    ScreenBlock = Screen(world.width, world.height)

    Wagent.set_y = 0
    Wagent.set_x = trial % world.width

    ScreenAgent.drawAgent_cumulative(Wagent)
    ScreenBlock.drawBlock_cumulative(block)

    print([Wagent.x, Wagent.y])

    for t in range(1, total_time):
        Wagent.x = ScreenAgent.wrapper(Wagent.x + motor_activity[t])

        if t < total_time:
            if block.direction == "right":
                block.x = ScreenBlock.wrapper(block.x + 1)
            else:
                block.x = ScreenBlock.wrapper(block.x - 1)

            Wagent.y = Wagent.y + 1
            block.y = block.y + 1

        ScreenAgent.drawAgent_cumulative(Wagent)
        ScreenBlock.drawBlock_cumulative(block)

    n_cols = 4
    n_rows = 1
    col = 0

    plt.subplot2grid((n_rows, n_cols), (0, col), rowspan=1, colspan=1)
    col += 1

    # plot block
    plt.imshow(ScreenBlock.screen, cmap=plt.cm.binary)
    plt.xlabel("x")
    plt.ylabel("y")
    win = "WON" if world._check_win(block, agent) else "LOST"
    direction = "━▶" if block.direction == "right" else "◀━"
    plt.title(
        "Game - Trial: {}, {}, {}, {}".format(block.size, block.type, direction, win)
    )

    # plot bot positions
    plt.subplot2grid((n_rows, n_cols), (0, col), rowspan=1, colspan=1)
    col += 1
    plt.imshow(ScreenAgent.screen, cmap=plt.cm.binary)
    plt.xlabel("x")
    plt.ylabel("y")

    # plot bot positions
    plt.subplot2grid((n_rows, n_cols), (0, col), rowspan=1, colspan=1)
    col += 1
    plt.imshow(ScreenAgent.screen + 2 * ScreenBlock.screen, cmap=plt.cm.magma)
    plt.xlabel("x")
    plt.ylabel("y")

    # plot states
    plt.subplot2grid((n_rows, n_cols), (0, col), rowspan=1, colspan=1)
    col += 1
    plt.imshow(agent.brain_activity[trial], cmap=plt.cm.binary)
    plt.yticks([])
    plt.xticks(range(agent.n_nodes), agent.node_labels)


# ----------------- Animations ------------------------------------------


def plot_trial_animation(agent, world, trial, block_patterns=None):

    ims = []
    for t in range(world.height):
        im = t

    anim = FuncAnimation(fig, animate, fargs=(line), frames=world.height, interval=200)

    return anim
