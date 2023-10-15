
import sys
import os
import re

# for text
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from sklearn.feature_extraction.text import TfidfVectorizer

# for nlp
import spacy
from spacy import displacy
from spacy.tokens import Span




POS_roles = {
    'subject': ['NOUN', 'PRON', 'PROPN'],
    'action': ['VERB'],
    'modifier': ['ADJ', 'ADV', 'AUX', 'DET', 'NUM', 'PART', 'SYM', 'PUNCT'],
    'connector': ['ADP', 'CONJ', 'SCONJ', 'CCONJ'],
    'misc': ['INTJ', 'SPACE'],
}

def get_maximal_spans(doc, POS_roles = POS_roles):
    # select root tokens, eg those that are central, non-trivial word,
    roots = [
        t for t in doc
        if t.pos_ in POS_roles['subject']
        and re.sub('[^a-zA-Z0-9]', '', t.text) != ''
    ]
    # collect associated subtree indices
    trees = {r.i: [t.i for t in r.subtree] for r in roots}
    
    # ignore non-maximal roots and associated subtree
    subtree_ids = [set(v) - {k,} for k, v in trees.items()]
    subtree_ids = [i for ids in subtree_ids for i in ids]
    trees = {k: v for k, v in trees.items() if k not in subtree_ids and v}
    
    # collect spans
    spans = [doc[min(v): max(v)+1] for v in trees.values()]
    return spans


# top-down
def get_dependency_paths_top_down(doc, allowed_pos = None, allowed_dep = None):
    '''
    Performs top-down exploration of the dependency tree of tokens parsed with spacy,
    and collect paths in the oriented tree, where token part-of-speeches meet 
    the 'allowed_pos' requirement and dependency meet the 'allowed_dep' requirement.

    Parameters
    ----------
    doc: Iterable.
    output of a spacy model applied on a text.

    allowed_pos: list, or None.
    list of the allowed part-of-speech tags, or None to allow all tags.

    allowed_dep: list, or None.
    list of the allowed dependency relations, or None to allow all relations.

    Returns
    -------
    completed: list.
    The retrieved list of paths meeting the requirements in the dependency tree
    '''
    completed = []
    temporary = [[t] for t in doc if (allowed_pos is None or t.pos_ in allowed_pos)]
    while temporary != []:
        checked = []
        for chain in temporary:
            chain_successors = [
                chain + [c] 
                for c in chain[-1].children 
                if (allowed_pos is None or c.pos_ in allowed_pos)
                and (allowed_dep is None or c.dep_ in allowed_dep)
            ]
            if chain_successors:
                checked += chain_successors
            else:
                completed.append(chain)
        temporary = checked
    return completed


# bottom-up (faster)
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


def merge_intersecting_paths(paths):
    '''
    Merge intersecting paths, or equivalently, paths sharing their root token.
    '''
    root2path = {}
    for path in paths:
        root = path[-1]
        if root in root2path:
            root2path[root].append(path)
        else:
            root2path[root] = [path]
    paths = [[t for p in ps for t in p] for ps in root2path.values()]
    return paths
    

def get_path_span(doc, path):
    imin = min([t.i for t in path])
    imax = max([t.i for t in path])
    span = doc[imin: imax+1]
    return span


def get_entity_spans(
    doc, 
    POS_roles = POS_roles, 
    extended_mwe = True, 
    max_distance = None,
    ):
    # compute list of dependency paths
    paths = get_dependency_paths_from_tree(
        doc, 
        max_distance,
        start_pos = POS_roles['subject'] + POS_roles['modifier'],
        trans_pos = POS_roles['subject'] + POS_roles['modifier'], 
        stop_pos = (None if extended_mwe else POS_roles['subject']),
        dep = None,
    )
    # only retain paths whose root is a central, non-trivial word
    paths = [
        p for p in paths 
        if p[-1].pos_ in POS_roles['subject']
        and re.sub('[^a-zA-Z0-9]', '', p[-1].text) != ''
    ]
    # fusion paths with common root, and sort by order of appearance
    paths = merge_intersecting_paths(paths)
    paths.sort(key = lambda path : path[-1].i, reverse = True)

    # collect the entity corresponding to each path
    spans = [get_path_span(doc, path) for path in paths]
    return (paths, spans)


def add_entity_spans_to_tokens(doc, spans_preds):
    for span, pred in spans_preds:
        try:
            #tokens.set_ents([Span(tokens, span.start, span.end, pred)])
            doc.ents += (Span(doc, span.start, span.end, pred),)
        except:
            print('Issue with "{}" from pos. {} to {}'.format(span.text, span.start, span.end))
    return doc


def shift_ents(doc, doc2, labels, verbose = False):
    shifted_ents = [
        (span.start_char, span.end_char, span.label_) 
        for span in doc2.ents if span.label_ in labels
    ]
    if verbose: 
        print(shifted_ents)
    shifted_ents = [
        (doc.char_span(start_idx = span[0], end_idx = span[1]), span[2]) 
        for span in shifted_ents
    ]
    doc = add_entity_spans_to_tokens(doc, shifted_ents)
    return doc
