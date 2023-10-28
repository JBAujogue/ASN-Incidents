import os, sys
import copy

# for data
import pandas as pd

# for viz
import streamlit as st
import plotly.express as px



#**********************************************************
#*                      functions                         *
#**********************************************************

# ------------------------- Paths -------------------------
path_to_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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




# ------------------------- Layout ------------------------
def center_text(text, thickness = 1, line_spacing = 1.) :
    '''
    Displays a text with centered indentation, with specified 
    thickness (the lower, the thickier).
    '''
    st.markdown(f"<h{thickness} style='text-align: center; line-height: {line_spacing}; color: black;'>{text}</h{thickness}>", unsafe_allow_html = True)
    return


def load_data(path_to_data):
    data = pd.read_excel(path_to_data)
    data.labels.fillna(value = '', inplace = True)
    return data

def load_labels_from_data():
    labels = sorted(list({l for ls in st.session_state.data.labels.apply(lambda ls: ls.split(';')) for l in ls}))
    labels = pd.DataFrame(labels, columns = ['Label'])
    labels['Class'] = None
    labels['Metaclass'] = None
    labels['All'] = None
    st.session_state.labels = labels
    return



#**********************************************************
#                     main script                         *
#**********************************************************
class TaxonomyPage:
    def compute_taxonomy_changes_from_label_changes(self):
        return
    
    def update_taxonomy(self):
        '''
        Insert changes in taxonomy of labels and export taxonomy.
        '''
        # st.session_state.taxonomy = copy.deepcopy(st.session_state.taxonomy_tmp)
        # save_taxonomy(st.session_state.taxonomy, 'file.json')
        return


    def display_taxonomy(self):
        st.subheader('Taxonomy')
        labels = st.data_editor(
            data = st.session_state.labels,
            num_rows = 'fixed',
            use_container_width = True,
        )

        fig = px.treemap(
            labels.dropna(), 
            path = ['All', 'Metaclass', 'Class', 'Label'],
        )
        fig.update_traces(root_color = "lightgrey")
        fig.update_layout(margin = dict(t = 50, l = 25, r = 25, b = 25))
        st.plotly_chart(fig)


        return
    
    def __call__(self):
        self.display_taxonomy()




class DataPage:
    def update_data(self, batch):
        st.session_state.batch = batch
        st.session_state.data.update(batch)
        return

    def update_batch(self):
        num = st.session_state.widget_batch_num
        size = st.session_state.batch_size
        st.session_state.batch_num = num
        st.session_state.batch = st.session_state.data[num: num + size]
        return
    
    def update_filter(self):
        # collect analyses in/out/all widgets
        # for each label:
        #   if 'in': take the subset of datapoints with 'label' contained in ";"-separated list of labels
        #   if 'out': take the subset of datapoints with 'label' not contained in ";"-separated list of labels
        return
    
    def display_batch(self):
        # display batche number
        num = st.number_input(
            label = 'Range of data', 
            min_value = 0, 
            max_value = st.session_state.max, 
            value = st.session_state.batch_num, 
            step = st.session_state.batch_size,
            key = 'widget_batch_num', 
            on_change = self.update_batch, 
        )
        # display batch content
        batch = st.data_editor(
            data = st.session_state.batch, 
            num_rows = 'fixed',
            use_container_width = True,
        )
        st.button('Save changes', on_click = self.update_data, kwargs = {'batch': batch})
        return
    
    def __call__(self):
        self.display_batch()



class App:
    def __init__(self, path_to_data, path_to_taxonomy):
        '''
        The static part of the user interface.
        '''
        self.path_to_data = path_to_data
        self.path_to_taxonomy = path_to_taxonomy
        self.pages = {
            'Manage Taxonomy and Labels': TaxonomyPage(),
            'Explore Data': DataPage(),
            #'Run Analysis': RunAnalysis(),
        }

    def init_session_state(self):
        batch_num = 0
        batch_size = 25

        st.session_state.data = load_data(self.path_to_data)
        # st.session_state.taxonomy = pd.read_excel(self.path_to_taxonomy)
        load_labels_from_data()

        st.session_state.filter = [True]*len(st.session_state.data)
        st.session_state.batch_num = batch_num
        st.session_state.batch_size = batch_size
        st.session_state.max = len(st.session_state.data)
        st.session_state.batch = st.session_state.data[batch_num: batch_num + batch_size]

        return

    def __call__(self):
        '''
        The app static part, used in order to navigates throughout the pages.
        ''' 
        center_text('Labeling tool', thickness = 1)
        st.write(' ')
        st.write(' ')
        st.sidebar.title('Navigation')

        # init session
        if 'data' not in st.session_state: 
            self.init_session_state()

        # select and run page
        key = 'selectbox_goto'
        page = st.sidebar.selectbox(
            label = 'Go to:', 
            options = ['-'] + list(self.pages.keys()),
        )
        if page in self.pages:
            self.pages[page]()








if __name__ == "__main__":
    st.set_page_config(page_title = 'Sanofi Paperless Explorer', layout = 'wide')
    app = App(
        os.path.join(path_to_data, 'labeled', 'headings', 'headings_data.xlsx'), 
        '',
    )
    app()

