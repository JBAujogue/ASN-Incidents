import os
from os.path import dirname, abspath
import sys
import copy

# for data
import pandas as pd
import numpy as np
import scipy

# for nlp
import spacy

# for viz
import streamlit as st
# from streamlit_agraph import agraph



#**********************************************************
#*                      functions                         *
#**********************************************************

# ------------------------- Paths -------------------------
path_to_repo = dirname(dirname(abspath(__file__)))
path_to_data = os.path.join(path_to_repo, 'data')
path_to_src  = os.path.join(path_to_repo, 'src')

sys.path.append(path_to_src)


# -------------------------- Src --------------------------
from tmtools.topic import get_spans_html
from tmtools.similarity import (
    filter_similarity_matrix,
    get_most_similar_indices,
    get_similarity_heatmap,
)
from tmtools.graph.graph_extraction import build_knowledge_graph_from_doc, get_quotient_graph
from tmtools.graph.graph_visualization import plot_graph_streamlit




# ------------------------- Layout ------------------------
def center_text(text, thickness = 1, line_spacing = 1.) :
    '''Displays a text with centered indentation, with specified thickness (the lower, the thickier)'''
    st.markdown("<h{} style='text-align: center; line-height: {}; color: black;'>{}</h{}>".format(thickness, line_spacing, text, thickness), unsafe_allow_html = True)
    return



#**********************************************************
#                     main script                         *
#**********************************************************

class App():
    def __init__(
        self, 
        path_to_data, 
        corpus_name, 
        text_topic_column, 
        span_topic_column,
        ):
        self.path_to_data = path_to_data
        self.corpus_name = corpus_name
        self.text_topic_column = text_topic_column
        self.span_topic_column = span_topic_column


    def init_session_state(self):
        st.session_state.doc_num = 1
        st.session_state.df_texts = pd.read_excel(
            os.path.join(self.path_to_data, 'processed', 'source_titles_topics.xlsx')
        )
        st.session_state.df_sents = pd.read_excel(
            os.path.join(self.path_to_data, 'processed', 'source_sentences.xlsx')
        )
        st.session_state.df_spans = pd.read_excel(
            os.path.join(self.path_to_data, 'processed', 'source_spans_topics.xlsx')
        )
        st.session_state.text_sim_matrix = scipy.sparse.load_npz(
            os.path.join(self.path_to_data, 'plots', 'd3graph', "sim_matrix_tfidf_texts.npz")
        )
        st.session_state.n_docs = st.session_state.df_texts.shape[0]

        ax = st.session_state.df_texts[self.text_topic_column].value_counts().plot.barh(figsize = (10, 20))
        ax.invert_yaxis()
        st.session_state.text_topic_plot = ax.get_figure()
        st.session_state.span_topic2color = {
            t: '#84bee8' for t in list(set(st.session_state.df_spans[self.span_topic_column].tolist()))
        }
        st.session_state.nlp = spacy.load('fr_dep_news_trf', exclude = ['ner'])
        return


    def sync_document(self):
        doc_id = st.session_state.doc_num-1
        st.session_state.head = '{} n° {} :'.format(self.corpus_name.capitalize(), doc_id+1)
        st.session_state.title = st.session_state.df_texts.at[doc_id, 'title']
        st.session_state.location = st.session_state.df_texts.at[doc_id, 'location']
        st.session_state.text = st.session_state.df_texts.at[doc_id, 'text']
        st.session_state.topic = st.session_state.df_texts.at[doc_id, self.text_topic_column]

        df_sents_doc = st.session_state.df_sents[st.session_state.df_sents.Doc_id == doc_id]
        df_spans_doc = st.session_state.df_spans[st.session_state.df_spans.Doc_id == doc_id]
        st.session_state.html = get_spans_html(
            df_sents_doc, 
            df_spans_doc, 
            st.session_state.nlp, 
            st.session_state.span_topic2color,
        )
        return
    

    def sync_similarity_heatmap(self):
        doc_id = st.session_state.doc_num-1
        small_sim_matrix, x_ids, y_ids = filter_similarity_matrix(
            st.session_state.df_texts, 
            st.session_state.text_sim_matrix, 
            key_column = 'location', 
            idx = doc_id,
        )
        st.session_state.hmap = get_similarity_heatmap(
            small_sim_matrix.toarray(), 
            x_labels = [x+1 for x in x_ids], 
            y_labels = [y+1 for y in y_ids], 
        )
        return
    

    def sync_graph(self):
        doc = st.session_state.nlp(st.session_state.text)
        graph = build_knowledge_graph_from_doc(doc)
        graph = get_quotient_graph(graph)
        st.session_state.graph = graph
        return
    

    def sync_variables(self, var_name):
        if var_name + '_widget' in st.session_state:
            st.session_state[var_name] = st.session_state[var_name + '_widget']
        self.sync_document()
        self.sync_similarity_heatmap()
        self.sync_graph()
        return

    
    def select_document(self):
        col_void0, col_doc, col_void1 = st.columns([3, 4, 3])
        with col_doc:
            doc_num = st.number_input(
                'Select {} to display (among {} {}s)'.format(
                    self.corpus_name,
                    st.session_state.n_docs,
                    self.corpus_name,
                ), 
                value = 1,
                min_value = 1, 
                max_value = st.session_state.n_docs + 1, 
                step = 1,
                key = 'doc_num_widget',
                on_change = self.sync_variables,
                args = ('doc_num', ),
            )
        return


    def display_classifications(self):
        with st.expander('Classification'):
            # display classifications
            col_topic, col_int, col_pred = st.columns(3)
            with col_topic: 
                center_text('Topic', thickness = 3)
                st.info(st.session_state.topic)
            
            # display text
            st.write(st.session_state.text)

            # display global statistics
            if st.checkbox('Display statistics of topics'):
                st.pyplot(st.session_state.text_topic_plot)
        return


    def display_ner(self):
        with st.expander('Named Entity Recognition'):
            st.markdown(st.session_state.html, unsafe_allow_html = True)
        return


    def display_similarities(self):
        # display documents similarity
        with st.expander('Text similarity'):
            center_text('Most similar {}s'.format(self.corpus_name), thickness = 3)
            col_void0, col_top, col_void1 = st.columns([3, 4, 3])
            with col_top: 
                n = st.number_input(
                    'number of {}s'.format(self.corpus_name), 
                    value = 1, 
                    min_value = 1, 
                    max_value = 20, 
                    step = 1,
                )
            topk_sims = get_most_similar_indices(
                st.session_state.text_sim_matrix, 
                n = n, 
                idx = st.session_state.doc_num-1,
            )
            for i, sim in topk_sims:
                header = '{} n° {}'.format(self.corpus_name.capitalize(), i+1) + '&nbsp; &nbsp; &nbsp; - &nbsp; &nbsp; &nbsp; {:.2f} % similarity'.format(sim * 100)
                st.write('***')
                center_text(header, thickness = 3)
                st.write(st.session_state.df_texts.at[i, 'text'])
            
            st.write('***')
            if st.checkbox('Display similarity heatmap'):
                st.markdown(
                    'This incident is located at **{}**, similarity is computed \
                    on incidents that occurred at the same place.'.format(st.session_state.location))
                st.pyplot(st.session_state.hmap)
        return


    def display_graph(self):
        # display graph representation
        with st.expander('Graph representation'):
            st.write('Entities have the form of agglomerated expressions, meaning that\
                    they consist of possibly several multi-word expressions that\
                    are syntactically connected and form a single overall notion.\
                    These agglomerated expressions are here separated into independent\
                    multi-word expressions, with semantic relations represented as a Graph.')
            return_value = plot_graph_streamlit(
                st.session_state.graph, 
                node_label = 'span', # 'id', 'word', 
                edge_label = 'text',
            )
        return

    
    def run(self):
        if 'df_texts' not in st.session_state: 
            self.init_session_state()
            self.sync_document()
            self.sync_similarity_heatmap()
            self.sync_graph()
        
        center_text('Nuclear Incidents UI', thickness = 1)
        st.write(' ')
        st.write(' ')

        self.select_document()
 
        center_text(st.session_state.head, thickness = 3)
        center_text(st.session_state.title, thickness = 3, line_spacing = 1.3)
        st.write(' ')
        st.write(' ')

        self.display_classifications()
        self.display_ner()
        self.display_similarities()
        self.display_graph()




if __name__ == "__main__":
    st.set_page_config(page_title = 'Nuclear Incidents UI', layout = 'wide')
    app = App(
        path_to_data, 
        corpus_name = 'incident', 
        text_topic_column = 'topic_NMF', 
        span_topic_column = 'topic_LSA',
    )
    app.run()

