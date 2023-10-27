
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

from ..span import POS_roles



#**********************************************************
#*                      functions                         *
#**********************************************************


# ------------------ KG Construction ----------------------
def build_knowledge_graph_from_doc(doc, POS_roles = POS_roles, doc_idx = 0):
    # build GOW
    graph = build_GOW(doc, doc_idx = doc_idx)
    
    # compute and add nominal expression to each subject & action
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
        attribute_name = 'span',
    )
    graph = add_attribute_spans(
        graph, 
        doc, 
        attribute_name = 'span',
        attribute_alt = 'word',
    )

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
    graph = remove_nodes(
        graph, 
        pos = POS_roles['modifier'] + POS_roles['connector'] + POS_roles['misc'],
    )
    return graph



def build_knowledge_graph_from_corpus(
    texts, 
    nlp, 
    POS_roles = POS_roles, 
    start_idx = 0,
    ):
    # fill graph
    graph = nx.MultiDiGraph()
    for i in tnrange(len(texts)):
        text = texts[i]
        doc = nlp(text)
        new_graph = build_knowledge_graph_from_doc(doc, POS_roles, doc_idx = i+start_idx)
        graph = nx.compose(graph, new_graph)
        
    # get quotient graph by merging nodes with same word and pos
    graph = get_quotient_graph(graph)
    return graph



# ---------------- Parsing on Spacy doc -------------------
def build_GOW(doc, POS_roles = POS_roles, doc_idx = 0):
    # init graph
    graph = nx.MultiDiGraph()
    nodes = []
    edges = []

    # create nodes
    word2count = {}
    token2node = {}
    for token in doc:
        if ((token.pos_ not in POS_roles['misc']) 
        and (re.sub('[,;:!\.\?\_\-\~\`\/\|\(\)\[\]\{\}]', '', token.text) != '')):
            word = token.text
            word2count[word] = 0 if word not in word2count else word2count[word]+1

            # treat different word occurences as different nodes
            node_id = '{}_{}_{}'.format(word, doc_idx, word2count[word])

            # create and store node
            node = {
                'id': node_id,
                'doc': doc_idx,
                'position': token.i,
                'word': word,
                'lemma': token.lemma_,
                'pos': token.pos_,
                'tag': token.tag_,
                'dep': token.dep_,
                'morphology': token.morph.to_dict(),
            }
            token2node[token] = node
            nodes.append((node_id, node))
    graph.add_nodes_from(nodes)

    # create edges
    for token in doc:
        head = token.head
        if token in token2node and head in token2node and head != token:
            # draw edge in reverse direction !
            s_id = token2node[token]['id']
            t_id = token2node[head]['id']

            if token.i < head.i:
                span = doc[token.i+1: head.i]
                orient = 'source -> target'
            else:
                span = doc[head.i+1: token.i]
                orient = 'target -> source'

            text = span.text
            lemma = span.lemma_
            dep = token.dep_

            # create edge
            edge = {
                's': s_id,
                't': t_id,
                'dep': dep,
                'text' : text,
                'lemma': lemma,
                'text orientation': orient,
            }
            edges.append((s_id, t_id, edge))
    graph.add_edges_from(edges)
    
    return graph


def get_dependency_paths_from_tree(
    doc, 
    max_distance = None, 
    min_length = 2,
    start_from_leaves = False,
    start_pos = None,
    trans_pos = None,
    stop_pos = None, 
    dep = None):
    '''
    Performs bottom-up exploration of the dependency tree of tokens parsed with spacy,
    and collect paths in the oriented tree, where token part-of-speeches meet 
    the requirements and dependency meet the 'dep' requirement,
    and that are maximal with respect to these requirements.

    Parameters
    ----------
    doc: Iterable.
    output of a spacy model applied on a text.

    start_pos: list, or None.
    list of the allowed part-of-speech tags at the beginning of paths, 
    or None to allow all tags.

    trans_pos: list, or None.
    list of the allowed part-of-speech tags during tree walk, 
    or None to allow all tags.

    stop_pos: list, or None.
    list of the allowed part-of-speech tags that stop the walk in the tree, 
    or None to allow all tags.

    dep: list, or None.
    list of the allowed dependency relations, or None to allow all relations.

    Returns
    -------
    completed: list.
    The retrieved list of paths meeting the requirements in the dependency tree
    '''
    completed = []
    temporary = [
        t for t in doc 
        if (start_pos is None or t.pos_ in start_pos)
        and not (start_from_leaves and set(t.children) & set(temporary))
    ]
    temporary = [[t] for t in temporary]
    
    while temporary != []:
        checked = []
        for chain in temporary:
            word = chain[-1]
            head = word.head
            
            # token and its head are different elements
            bool1 = (word.dep_ != 'ROOT')
            
            # distance between last token and its head isn't too high
            bool2 = (max_distance is None or abs(word.i - head.i) <= max_distance) 
            
            # dependency fulfills criterion
            bool3 = (dep is None or word.dep_ in dep)
            
            # head fulfills transition criterion
            bool4 = (trans_pos is None or head.pos_ in trans_pos)
            
            # last token doesn't fulfills stopping criterion
            bool5 = len(chain) == 1 or not (stop_pos and word.pos_ in stop_pos)
            
            # accumulation step
            if (bool1 and bool2 and bool3 and bool4 and bool5):
                checked.append(chain + [head])
            else:
                completed.append(chain)
        temporary = checked

    completed = [c for c in completed if len(c) >= min_length]
    return completed



# ------------------ Graph operations ---------------------
def get_dependency_paths_from_graph(
    graph, 
    max_distance = None, 
    min_length = 2,
    start_from_leaves = False,
    start_pos = None,
    trans_pos = None,
    stop_pos = None, 
    dep = None):
    '''
    Performs bottom-up exploration of a dependency graph,
    and collect paths in the oriented graph, where token part-of-speeches meet 
    the requirements and dependency meet the 'dep' requirement,
    and that are maximal with respect to these requirements.

    Parameters
    ----------
    graph: nx.DiGraph object.

    start_pos: list, or None.
    list of the allowed part-of-speech tags at the beginning of paths, 
    or None to allow all tags.

    trans_pos: list, or None.
    list of the allowed part-of-speech tags during tree walk, 
    or None to allow all tags.

    stop_pos: list, or None.
    list of the allowed part-of-speech tags that stop the walk in the tree, 
    or None to allow all tags.

    dep: list, or None.
    list of the allowed dependency relations, or None to allow all relations.

    Returns
    -------
    completed: list.
    The retrieved list of paths meeting the requirements in the dependency graph
    '''
    
    completed = []
    temporary = [
        n for n in graph.nodes 
        if (start_pos is None or graph.nodes[n]['pos'] in start_pos)
        and (not start_from_leaves or list(graph.predecessors(n)) == [])
    ]
    temporary = [[t] for t in temporary]
    
    while temporary != []:
        checked = []
        for chain in temporary:
            word = chain[-1]
            head = list(graph.successors(word))
            if head: head = head[0]
            
            # token has a head
            bool1 = (head != [])
            
            # distance between last token and its head isn't too high
            bool2 = bool1 and (max_distance is None or abs(graph.nodes[word]['position'] - graph.nodes[head]['position']) <= max_distance) 
            
            # dependency fulfills criterion
            bool3 = (dep is None or graph.nodes[word]['dep'] in dep)
            
            # head fulfills transition criterion
            bool4 = bool1 and (trans_pos is None or graph.nodes[head]['pos'] in trans_pos)
            
            # last token doesn't fulfills stopping criterion
            bool5 = len(chain) == 1 or not (stop_pos and graph.nodes[word]['pos'] in stop_pos)
            
            # accumulation step
            if (bool1 and bool2 and bool3 and bool4 and bool5):
                checked.append(chain + [head])
            else:
                completed.append(chain)
        temporary = checked
        
    completed = [c for c in completed if len(c) >= min_length]
    return completed


def revert_edge_orientation(graph, dep_to_revert):
    new_graph = copy.deepcopy(graph)
    
    for e in graph.edges:
        if graph.edges[e]['dep'] in dep_to_revert:
            # create new edge
            new_e = dict(graph.edges[e])
            s, t = new_e['s'], new_e['t']
            new_e['s'] = t
            new_e['t'] = s
            
            # replace old edge by newly created one
            new_graph.remove_edge(s, t)
            new_graph.add_edges_from([(t, s, new_e)])
    return new_graph


def remove_nodes(graph, pos):
    graph = copy.deepcopy(graph)
    temps = [n for n in graph.nodes if graph.nodes[n]['pos'] in pos]
    for n in temps:
        graph.remove_node(n)
    return graph



# -------------- Attributes and Relations -----------------
def convert_paths_to_attributes(graph, paths, attribute_name, remove_paths = False):
    graph = copy.deepcopy(graph)
    nouns = set(list([p[-1] for p in paths]))
    attrs = set(list([w for p in paths for w in p[:-1]]))
    noun2path = {n: list([p[:-1] for p in paths if p[-1] == n]) for n in nouns}
    
    # add dependency paths as attribute to central nodes
    for noun in nouns:
        graph.nodes[noun][attribute_name] = noun2path[noun]
    
    # remove dependency paths from graph
    if remove_paths:
        for attr in attrs:
            graph.remove_node(attr)
    return graph



def add_attribute_spans(graph, doc, attribute_name, attribute_alt = None):
    graph = copy.deepcopy(graph)
    for n in graph.nodes:
        node = graph.nodes[n]
        if attribute_name in node:
            ids = [n] + [m for p in node[attribute_name] for m in p]
            ids = [graph.nodes[m]['position'] for m in ids]
            imin = min(ids)
            imax = max(ids)
            span = doc[imin: imax+1].text
            graph.nodes[n][attribute_name] = span
        elif attribute_alt in node:
            graph.nodes[n][attribute_name] = graph.nodes[n][attribute_alt]
    return graph


def convert_paths_to_relations(graph, paths, remove_paths = False):
    def path_to_relation(path):
        return list(path[1:-1])
    
    graph = copy.deepcopy(graph)
    edges = [(p[0], p[-1], tuple(path_to_relation(p))) for p in paths]
    graph.add_edges_from(edges)
        
    if remove_paths:
        nodes = set([p[i] for p in paths for i in [0, -1]])
        temps = set([n for p in paths for n in p[1:-1]]) - nodes
        nodes = list(nodes)
        temps = list(temps)
        for temp in temps:
            graph.remove_node(temp)
    return graph



# ------------------- Graph Quotient ----------------------
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
            'span': S.nodes[node]['span'],
        }
        return data
    
    def equiv_classes_edge(e, f):
        data = {
            # custom
            'text': graph.edges(e, f)['word'],
            'pos': S.nodes[node]['pos'],
            'span': S.nodes[node]['span'],
        }
        return data
    
    graph = nx.quotient_graph(
        graph, 
        partition = equiv_relation, 
        node_data = equiv_classes_node,
        edge_data = equiv_classes_edge,
    )
    return graph