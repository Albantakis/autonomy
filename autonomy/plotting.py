import networkx as nx
import numpy as np

from .structuralAgentAnalysis import densely_connected_nodes
from .utils import *


def plot_animat_brain(agent, state=None, ax=None):
    n_nodes = agent.n_nodes
    G = get_graph(agent)

    node_type = np.array([0 for i in range(n_nodes)])
    node_type[agent.motor_ixs] = 2
    node_type[agent.hidden_ixs] = 1

    # set colors according to state and node type (sensor, motor, hidden)
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
            [red_off, green_off, blue_off, grey_off, white],
            [red, green, blue, grey, white],
        ]
    )

    state = [1] * n_nodes if state == None else state

    node_colors = [colors[state[i], node_type[i]] for i in range(n_nodes)]
    node_labels = np.array(G.nodes)

    connected_nodes = (
        list(densely_connected_nodes(agent.cm)) + agent.motor_ixs + agent.sensor_ixs
    )
    node_colors = [
        node_colors[i] if i in connected_nodes else colors[state[i], 3]
        for i in range(n_nodes)
    ]

    isolates = [x for x in nx.isolates(G)]
    node_colors = [
        node_colors[i] if node_labels[i] not in isolates else colors[0, 4]
        for i in range(n_nodes)
    ]

    # indicate self-loops by thick lines
    self_nodes_ixs = [i for i in range(n_nodes) if agent.cm[i, i] == 1]
    linewidths = [2.5 if i in self_nodes_ixs else 1 for i in range(n_nodes)]

    # position
    n_rows = int(np.ceil(agent.n_hidden / 2)) + 2

    subset_type = np.array([0 for i in range(n_nodes)])
    subset_type[agent.motor_ixs] = n_rows - 1
    for c, i in enumerate(agent.hidden_ixs):
        subset_type[i] = 1 + int(c / 2)

    for c, n in enumerate(G.nodes):
        G.nodes[n]["subset"] = subset_type[c]

    pos = nx.drawing.layout.multipartite_layout(G, align="horizontal")

    diff_x = 0.5
    if len(agent.sensor_ixs) > 2:
        for c, n in enumerate(node_labels[agent.sensor_ixs]):
            (x, y) = pos[n]
            if (c + 1) % 2 == 0:
                pos[n] = (diff_x - int(c / 2) - diff_x / 2, y)
            else:
                pos[n] = (diff_x - int(c / 2) + diff_x / 2, y)

    for c, n in enumerate(node_labels[agent.hidden_ixs]):
        (x, y) = pos[n]
        if ((c + 1) % 2) == 1:
            pos[n] = (x + (int(c / 4) + 1) * diff_x * 1.2, y)
        else:
            pos[n] = (x - (int(c / 4) + 1) * diff_x * 1.2, y)

    flipped_pos = {node: (-x, -y) for (node, (x, y)) in pos.items()}

    nx.draw(
        G,
        with_labels=True,
        node_size=800,
        node_color=node_colors,
        edgecolors="#000000",
        linewidths=linewidths,
        pos=flipped_pos,
        ax=ax,
    )
