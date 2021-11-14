# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division
import os
from os.path import dirname, abspath
import sys
from io import open
import unicodedata
import string
import re
import random
import itertools
import copy
import dill

# for data
import pandas as pd
import numpy as np

# for nlp
import spacy
from spacy.tokens import Span
from spacy import displacy

# for viz
from PIL import Image
from termcolor import colored
import altair as alt
import streamlit as st
from streamlit_agraph import agraph



#**********************************************************
#*                      functions                         *
#**********************************************************

# ------------------------- Paths -------------------------
path_to_repo = dirname(dirname(abspath(__file__)))
path_to_data = os.path.join(path_to_repo, 'data')
path_to_save = os.path.join(path_to_repo, 'saves')
path_to_src  = os.path.join(path_to_repo, 'src')

sys.path.append(path_to_src)


# -------------------------- Src --------------------------
from TextSimilarity import Corrélateur
from EntityClassification import unsupervizedEntityClassifier, load_entity_classifier
from EntityExtraction import shift_ents
from EntityVisualization import get_and_display_entities, get_html
from GraphExtraction import build_knowledge_graph_from_doc, get_quotient_graph
from GraphVisualization import plot_graph_streamlit




# ------------------------- Layout ------------------------
def centerText(text, thickness = 1, line_spacing = 1.) :
    '''Displays a text with centered indentation, with specified thickness (the lower, the thickier)'''
    st.markdown("<h{} style='text-align: center; line-height: {}; color: black;'>{}</h{}>".format(thickness, line_spacing, text, thickness), unsafe_allow_html = True)
    return



#**********************************************************
#                     main script                         *
#**********************************************************
st.set_page_config(page_title = 'Nuclear Incidents UI', layout = 'wide')
centerText('Nuclear Incidents UI', thickness = 1)
st.write(' ')
st.write(' ')



# session state
if 'data' not in st.session_state:
    st.session_state.data = None
    st.session_state.doc_num = None
    st.session_state.doc_idx = None
    st.session_state.head = None
    st.session_state.title = None
    st.session_state.text = None
    st.session_state.location = None
    st.session_state.topic_base = None
    st.session_state.topic_interpretated = None
    st.session_state.topic_predicted = None
    st.session_state.doc = None

    st.session_state.text_html = None
    st.session_state.doc_html = None
    st.session_state.doc_html_inter = None
    st.session_state.graph = None

    st.session_state.text_classifier = None
    st.session_state.entity_classifier_base = None
    st.session_state.entity_classifier_interpretated = None
    st.session_state.correlateur = None
    st.session_state.nlp = None
    #st.session_state.nlp_en = None

    st.session_state.doc_name = 'incident'
    st.session_state.n_docs = None
    st.session_state.show_every = 1

if st.session_state.data is None : 
    data = pd.read_excel(
        os.path.join(path_to_data, 'data_nuclear_incidents_with_titles_topics.xlsx'), 
        engine = 'openpyxl',
        header = 0)
    st.session_state.data = data
    st.session_state.n_docs = data.shape[0]
	
    st.session_state.nlp = spacy.load('fr_dep_news_trf', disable = ['ner'])
    #st.session_state.nlp_en = spacy.load('en_core_web_trf')

    path_to_clf = os.path.join(path_to_save, 'text_classifier.pk')
    with open(path_to_clf, 'rb') as file: 
        st.session_state.text_classifier = dill.load(file)

    path_to_clf = os.path.join(path_to_save, 'NMF_entity_classifier_base.pk')
    with open(path_to_clf, 'rb') as file: 
        st.session_state.entity_classifier_base = dill.load(file)

    path_to_clf = os.path.join(path_to_save, 'NMF_entity_classifier_interpretated.pk')
    with open(path_to_clf, 'rb') as file: 
        st.session_state.entity_classifier_interpretated = dill.load(file)

    st.session_state.correlateur = Corrélateur(data, text_column = 'text', key_column = 'location')



# select document
col_void0, col_doc, col_void1 = st.columns([3, 4, 3])
with col_doc:
    doc_num = st.number_input(
        'Select {} to display (among {} {}s)'.format(
            st.session_state.doc_name,
            st.session_state.n_docs,
            st.session_state.doc_name,
        ), 
        value = 1,
        min_value = 1, 
        max_value = st.session_state.n_docs + 1, 
        step = st.session_state.show_every,
    )
    st.write(' ')
    st.write(' ')
    if doc_num != st.session_state.doc_num:
        doc_idx = doc_num-1
        text = st.session_state.data.text.tolist()[doc_idx]

        # load document
        st.session_state.doc_num = doc_num
        st.session_state.doc_idx = doc_idx
        st.session_state.text = text
        st.session_state.head = '{} n° {} :'.format('Incident', doc_num)
        st.session_state.title = st.session_state.data.title.tolist()[doc_idx]
        st.session_state.location = st.session_state.data.location.tolist()[doc_idx]
        st.session_state.topic_base = st.session_state.data.topic_NMF.tolist()[doc_idx]
        st.session_state.topic_interpretated = st.session_state.data.topic_NMF_interpreted.tolist()[doc_idx]
        st.session_state.topic_predicted = st.session_state.text_classifier.predict([text])[0]

        # perform NLU
        doc = st.session_state.nlp(text)
        #doc_en = st.session_state.nlp_en(text)
        #doc = shift_ents(doc, doc_en, labels = ['GPE', 'DATE'])

        # compute htmls
        st.session_state.text_html = get_html(text)
        st.session_state.doc_html = get_and_display_entities(
            doc = copy.deepcopy(doc),
            clf = st.session_state.entity_classifier_base,
            extended_mwe = True, 
            max_distance = None,
            threshold = 0.05, #threshold, 
            color = '#84bee8', # blue
        )
        st.session_state.doc_html_inter = get_and_display_entities(
            doc = copy.deepcopy(doc),
            clf = st.session_state.entity_classifier_interpretated,
            extended_mwe = True, 
            max_distance = None,
            threshold = 0.05, # threshold,
            color = '#ff6e70', # red
        )
        graph = build_knowledge_graph_from_doc(doc)
        graph = get_quotient_graph(graph)
        st.session_state.graph = graph



# display heading
centerText(st.session_state.head, thickness = 3)
centerText(st.session_state.title, thickness = 3, line_spacing = 1.3)
st.write(' ')
st.write(' ')



# display documents with classification
with st.expander('Classification'):
    col_base, col_int, col_pred = st.columns(3)
    with col_base: 
        centerText('Automatic class', thickness = 3)
        st.info(st.session_state.topic_base)
    with col_int: 
        centerText('Interpreted class', thickness = 3)
        st.info(st.session_state.topic_interpretated)
    with col_pred: 
        centerText('Predicted class', thickness = 3)
        st.success(st.session_state.topic_predicted)
    
    st.markdown(st.session_state.text_html, unsafe_allow_html = True)

    if st.checkbox('Display Automatically formed classes'):
        ax = st.session_state.data.topic_NMF.value_counts().plot.barh(figsize = (10, 20))
        ax.invert_yaxis()
        fig = ax.get_figure()
        st.pyplot(fig)

    if st.checkbox('Display Interpreted classes'):
        ax = st.session_state.data.topic_NMF_interpreted.value_counts().plot.barh(figsize = (10, 10))
        ax.invert_yaxis()
        fig = ax.get_figure()
        st.pyplot(fig)



# display document with detected entities
with st.expander('Named Entity Recognition'):
    ent_type = st.radio('', options = [
        'Automatically formed entities',
        'Interpreted entities', 
    ])
    if ent_type == 'Automatically formed entities': # (unsupervised recognition)
        #threshold = st.slider('Threshold:', min_value = 0., max_value = 1., value = 0.15, step = 0.05)
        st.markdown(st.session_state.doc_html, unsafe_allow_html = True)

    elif ent_type == 'Interpreted entities': # (supervised recognition)
        #threshold = st.slider('Threshold:', min_value = 0., max_value = 1., value = 0.15, step = 0.05)
        st.markdown(st.session_state.doc_html_inter, unsafe_allow_html = True)



# display documents similarity
with st.expander('Text similarity'):
    st.markdown('This incident is located at **{}**, similarity is computed \
              on incidents that occurred at the same place.'.format(st.session_state.location))
    if st.checkbox('Display similarity grid'):
        fig = st.session_state.correlateur.get_correlations_plot(key1 = st.session_state.location)
        st.pyplot(fig)

    centerText('Most similar incidents', thickness = 3)
    col_void0, col_top, col_void1 = st.columns([3, 4, 3])
    with col_top: 
        topk = st.number_input('number of incidents', value = 0, min_value = 0, max_value = 10, step = 1)
    if topk > 0:
        (idxs, texts, sims) = st.session_state.correlateur.get_most_similar(
            incident_num = st.session_state.doc_num,
            topk = topk,
        )
        for idx, text, sim in zip(idxs, texts, sims):
            header = 'Incident n° ' + str(idx) + '&nbsp; &nbsp; &nbsp; - &nbsp; &nbsp; &nbsp; {:.2f} % similarity'.format(sim * 100)
            centerText(header, thickness = 3)
            text_html = get_html(text)
            st.markdown(text_html, unsafe_allow_html = True)
    


# display graph representation
with st.expander('Graph representation'):
    st.write('Entities have the form of agglomerated expressions, meaning that\
              they consist of possibly several multi-word expressions that\
              are syntactically connected and form a single overall notion.\
              These agglomerated expressions are here separated into independent\
              multi-word expressions, with semantic relations represented as a Graph.')

    return_value = plot_graph_streamlit(
        st.session_state.graph, 
        node_label = 'description span', # 'id', 'word', 
        edge_label = 'text',
    )

