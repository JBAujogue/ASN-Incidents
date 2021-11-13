
import sys
import os

# for data
import pandas as pd
import numpy as np

# for nlp
import spacy
from spacy import displacy
from spacy.tokens import Span

# custom
from EntityExtraction import get_entity_spans, add_entity_spans_to_tokens




# taken from https://github.com/explosion/spacy-streamlit/blob/9592a27645f9bdb0c02c6add02838a506a0aaccf/spacy_streamlit/util.py#L26
def get_html(html: str):
    """Convert HTML so it can be rendered."""
    WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
    # Newlines seem to mess with the rendering
    html = html.replace("\n", " ")
    return WRAPPER.format(html)



def get_and_display_entities(
    doc, 
    clf, 
    extended_mwe = True, 
    max_distance = None,
    threshold = .1,
    color = '#84bee8', # blue
    ):
    # extract entities
    paths, spans = get_entity_spans(doc, extended_mwe = True, max_distance = None)

    # classify entities
    texts = [ent.lemma_ for ent in spans]
    preds = clf.predict(texts)
    spans_preds = [(e, p[0]) for e, p in zip(spans, preds) if p[1]>= threshold]
    
    # get text with entity spans and predicted classes as html
    doc = add_entity_spans_to_tokens(doc, spans_preds)
    
    # display html
    topic2color = {topic: color for topic in clf.topic_dict.values()}
    doc_html = displacy.render(
        doc, 
        style = "ent", 
        jupyter = False,
        options = {'colors': topic2color},
    )
    doc_html = get_html(doc_html)
    return doc_html