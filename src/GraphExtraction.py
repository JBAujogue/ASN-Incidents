
import sys
import os
import copy
import re

# for data
import pandas as pd
import numpy as np

# for text
import spacy

# for graphs
import networkx as nx

# for viz
from tqdm import tnrange

# custom
from GraphUtils import (
    POS_roles,
    get_dependency_graph,
    get_dependency_paths_from_graph,
    convert_paths_to_attributes,
    add_attribute_spans,
    convert_paths_to_relations,
    remove_nodes,
    equiv_relation,
    equiv_classes_data,
)


#**********************************************************
#*                      functions                         *
#**********************************************************


# ------------------ KG Construction ----------------------
def build_knowledge_graph_from_doc(doc, doc_idx = 0):
    # build GOW
    graph = get_dependency_graph(doc, doc_idx = doc_idx)
    
    # compute and merge subject attributes
    attributes = get_dependency_paths_from_graph(
        graph, 
        max_distance = None, 
        min_length = 2,
        start_from_leaves = True,
        start_pos = POS_roles['modifier'] + POS_roles['connector'] + POS_roles['misc'],
        trans_pos = None, 
        stop_pos  = POS_roles['subject'] + POS_roles['action'],
        dep = None,
    )
    graph = convert_paths_to_attributes(
        graph, 
        attributes, 
        attribute_name = 'description',
    )
    # add text span of each subject
    graph = add_attribute_spans(
        graph, 
        doc, 
        attribute_name = 'description',
        attribute_alt = 'word',
    )
    # compute and merge wider subject attributes
    wide_attributes = get_dependency_paths_from_graph(
        graph, 
        max_distance = None, 
        min_length = 2,
        start_from_leaves = True,
        start_pos = POS_roles['subject'] + POS_roles['modifier'],
        trans_pos = POS_roles['subject'] + POS_roles['modifier'],
        stop_pos  = None,
        dep = None,
    )
    wide_attributes = [
        p for p in wide_attributes 
        if graph.nodes[p[-1]]['pos'] in POS_roles['subject']
        and re.sub('[^a-zA-Z0-9]', '', graph.nodes[p[-1]]['word']) != ''
    ]
    graph = convert_paths_to_attributes(graph, wide_attributes, attribute_name = 'wide description')
    
    # add wider text span of each subject
    graph = add_attribute_spans(graph, doc, attribute_name = 'wide description')
    
    # compute and merge relations
    relations = get_dependency_paths_from_graph(
        graph, 
        max_distance = None, 
        start_from_leaves = False,
        start_pos = POS_roles['subject'],
        trans_pos = None, 
        stop_pos  = POS_roles['subject'] + POS_roles['action'],
        min_length = 3,
        dep = None,
    )
    graph = convert_paths_to_relations(graph, relations)
    
    # remove uneccessary nodes
    graph = remove_nodes(graph, pos = POS_roles['modifier'] + POS_roles['connector'] + POS_roles['misc'])
    return graph


def get_quotient_graph(graph):
    '''
    Get quotient graph by merging nodes with same word and pos.
    '''
    def equiv_relation(u, v):
        same_word = graph.nodes[u]['word'] == graph.nodes[v]['word'] 
        same_pos  = graph.nodes[u]['pos'] == graph.nodes[v]['pos']
        return (same_word and same_pos)

    def equiv_classes_node(b):
        S = graph.subgraph(b)
        nodes = list(S.nodes)
        nodes.sort(key = lambda n: int(n.split('_')[-1]))
        node = nodes[0]
        data = {
            # default
            'graph': S, 
            'nnodes': len(S), 
            'nedges': S.number_of_edges(), 
            'density': nx.density(S),

            # custom
            'id': S.nodes[node]['word'] + '_' + S.nodes[node]['pos'],
            'word': S.nodes[node]['word'],
            'pos': S.nodes[node]['pos'],
            'description span': S.nodes[node]['description span'],
        }
        return data
    
    def equiv_classes_edge(e, f):

        data = {
            # custom
            'text': graph.edges(e, f)['word'],
            'pos': S.nodes[node]['pos'],
            'description span': S.nodes[node]['description span'],
        }
        return data
    
    graph = nx.quotient_graph(
        graph, 
        partition = equiv_relation, 
        node_data = equiv_classes_node,
        edge_data = equiv_classes_edge,
    )
    return graph



def build_knowledge_graph_from_corpus(texts, nlp, start_idx = 0):
    # fill graph
    graph = nx.MultiDiGraph()
    for i in tnrange(len(texts)):
        text = texts[i]
        doc = nlp(text)
        new_graph = build_knowledge_graph_from_doc(doc, doc_idx = i+start_idx)
        graph = nx.compose(graph, new_graph)
        
    # get quotient graph by merging nodes with same word and pos
    graph = get_quotient_graph(graph)
    return graph