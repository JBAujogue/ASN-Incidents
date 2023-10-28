import sys
import os
import copy

# for data
import pandas as pd
import numpy as np

# for text
import spacy

# for graphs
import networkx as nx

# for viz
import matplotlib.pyplot as plt
from streamlit_agraph import agraph, Node, Edge, Config



#**********************************************************
#*                      functions                         *
#**********************************************************

def plot_graph(graph, node_attribute = 'word'):
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(graph)
    nx.draw(
        graph, 
        pos,
        node_color = '#00b4d9', 
        node_size = 50, 
        width = 0.25,
        edge_cmap = plt.cm.Blues, 
    )
    node_labels = nx.get_node_attributes(graph, node_attribute)
    nx.draw_networkx_labels(graph, pos, labels = node_labels)
    plt.show()


# see https://blog.streamlit.io/the-streamlit-agraph-component/
def plot_graph_streamlit(graph, node_label = 'word', edge_label = 'text'):
    pos2color = {
        'NOUN': '#4d9dd6', # blue
        'PRON': '#4d9dd6', # blue
        'PROPN':'#4d9dd6', # blue
        'VERB': '#ff7f3b', # orange
    }
    nodes = [
        Node(id = graph.nodes[n]['id'],
             label = graph.nodes[n][node_label],
             size = 15,
             color = (pos2color[graph.nodes[n]['pos']]) if graph.nodes[n]['pos'] in pos2color else '#50ebb2')
        for n in graph.nodes
    ]
    edges = [
        Edge(source = graph.nodes[e[0]]['id'],
             target = graph.nodes[e[1]]['id'],
             #label = graph.edges[e][edge_label],
             #type = "CURVE_SMOOTH"
            )
        for e in graph.edges
    ]
    config = Config(
        width = 2000, 
        height = 800, 
        directed = True,
        physics = True, 
        nodeHighlightBehavior = True, 
        highlightColor = "#F7A7A6", # or "blue"
        collapsible = True,
        node = {'labelProperty':'label'},
        link = {'labelProperty': 'label', 'renderLabel': True},
    ) 
    return_value = agraph(nodes = nodes, edges = edges, config = config)
    return return_value


