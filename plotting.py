import numpy as np
import networkx as nx
import copy

#from matplotlib import pyplot as plt

def plot_animat_brain(cm, graph=None, state=None, ax=None):
    '''
    Function description
        Inputs:
            inputs:
        Outputs:
            outputs:
    '''

    n_nodes = cm.shape[0]
    if n_nodes==7:
        labels = ['S1','M1','M2','A','B','C','D']
        pos = {'S1': (5,40), #'S2': (20, 40),
           'A': (0, 30), 'B': (20, 30),
           'C': (0, 20), 'D': (20, 20),
          'M1': (5,10), 'M2': (15,10)}
        nodetype = (0,1,1,2,2,2,2)

        ini_hidden = 3

    elif n_nodes==8:
        labels = ['S1','S2','M1','M2','A','B','C','D']
        pos = {'S1': (5,40), 'S2': (15, 40),
           'A': (0, 30), 'B': (20, 30),
           'C': (0, 20), 'D': (20, 20),
          'M1': (5,10), 'M2': (15,10)}
        nodetype = (0,0,1,1,2,2,2,2)
        ini_hidden = 4

    if graph is None:
        graph = nx.from_numpy_matrix(cm, create_using=nx.DiGraph())
        mapping = {key:x for key,x in zip(range(n_nodes),labels)}
        graph = nx.relabel_nodes(graph, mapping)

    state = [1]*n_nodes if state==None else state

    blue, red, green, grey, white = '#6badf9', '#f77b6c', '#8abf69', '#adadad', '#ffffff'
    blue_off, red_off, green_off, grey_off = '#e8f0ff','#ffe9e8', '#e8f2e3', '#f2f2f2'

    colors = np.array([[red_off,blue_off,green_off, grey_off, white],
                       [red,blue,green, grey, white]])

    node_colors = [colors[state[i],nodetype[i]] for i in range(n_nodes)]
    # Grey Uneffective or unaffected nodes
    cm_temp = copy.copy(cm)
    cm_temp[range(n_nodes),range(n_nodes)]=0
    unaffected = np.where(np.sum(cm_temp,axis=0)==0)[0]
    uneffective = np.where(np.sum(cm_temp,axis=1)==0)[0]
    noeffect = list(set(unaffected).union(set(uneffective)))
    noeffect = [ix for ix in noeffect if ix in range(ini_hidden,ini_hidden+4)]
    node_colors = [node_colors[i] if i not in noeffect else colors[state[i],3] for i in range(n_nodes)]

    #   White isolate nodes
    isolates = [x for x in nx.isolates(graph)]
    node_colors = [node_colors[i] if labels[i] not in isolates else colors[0,4] for i in range(n_nodes)]

    self_nodes = [labels[i] for i in range(n_nodes) if cm[i,i]==1]
    linewidths = [2.5 if labels[i] in self_nodes else 1 for i in range(n_nodes)]

#     fig, ax = plt.subplots(1,1, figsize=(4,6))
    nx.draw(graph, with_labels=True, node_size=800, node_color=node_colors,
    edgecolors='#000000', linewidths=linewidths, pos=pos, ax=ax)