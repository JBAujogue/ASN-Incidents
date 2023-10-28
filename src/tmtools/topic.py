# for data 
import numpy as np
import pandas as pd

# for nlp
from spacy.tokens import Span

# for topic modeling
from sklearn.decomposition import PCA, NMF, LatentDirichletAllocation, TruncatedSVD

# for visualization
import matplotlib.pyplot as plt
from spacy import displacy




def compute_topic_modeling(
    tfidf_matrix, 
    tfidf_ngrams,
    method = 'LSA', 
    n_components = None,
    topic_name = 'Topic',
    ):
    if method not in ['PCA', 'LSA', 'LDA', 'NMF']:
        print('Invalid method (choose between PCA, LSA, LDA and NMF)')
        return
    
    # set model
    if method == 'PCA' : 
        model = PCA(n_components = n_components, random_state = 42)
        tfidf_matrix = tfidf_matrix.todense()
        
    elif method == 'LSA' : 
        model = TruncatedSVD(
            n_components = (n_components if n_components else min(tfidf_matrix.shape)-1), 
            n_iter = 10,
            random_state = 42,
        )
        
    elif method == 'LDA' : 
        model = LatentDirichletAllocation(
            n_components = (n_components if n_components else min(tfidf_matrix.shape)-1), 
            learning_offset = 50.,
            random_state = 42,
        )
        
    elif method == 'NMF':
        model = NMF(
            n_components = (n_components if n_components else min(tfidf_matrix.shape)-1), 
            beta_loss = 'kullback-leibler', 
            solver = 'mu', 
            random_state = 42,
        )
    
    # compute (text * topic) matrix
    text_topic = model.fit_transform(tfidf_matrix)
    topics = ['{} {}'.format(topic_name, i+1) for i in range(text_topic.shape[1])]
    df_text_topic = pd.DataFrame(text_topic, columns = topics)

    # compute (topic * feature) matrix
    df_topic_feature = pd.DataFrame(np.round(model.components_, 4))
    df_topic_feature.columns = tfidf_ngrams
    df_topic_feature.index = topics
    
    # compute topic importance
    try:
        eigenvalues = model.explained_variance_ratio_ #model.singular_values_ 
    except:
        eigenvalues = [1. for _ in range(model.components_.shape[0])]
    df_topic_importance = pd.DataFrame({'topic' : topics, 'weight': eigenvalues})
    
    # compute feature importance
    feature_importance = abs(df_topic_feature.T).dot(eigenvalues)
    df_feature_importance = pd.DataFrame({'feature': tfidf_ngrams, 'weight' : feature_importance})
    
    return (df_text_topic, df_topic_feature, df_topic_importance, df_feature_importance)



# see https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html
def plot_topic_words(df_topic_feature, n_topics, n_top_words = 15, title = ''):
    feature_names = df_topic_feature.columns
    n_row_plots = int((n_topics-0.1)/3)+1
    fig, axes = plt.subplots(n_row_plots, 3, figsize = (25, 8*n_row_plots), sharex = True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(df_topic_feature.values[:n_topics,:]):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx +1}', fontdict = {'fontsize': 30})
        ax.invert_yaxis()
        ax.tick_params(axis = 'both', which = 'major', labelsize = 20)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize = 40)

    plt.subplots_adjust(top = 0.90, bottom = 0.05, wspace = 0.90, hspace = 0.3)
    return fig



def get_topic_description(df, n_topic = 1, limit = 10) :
    features = df.columns.tolist()
    topic = df.iloc[n_topic-1, :]
    if isinstance(limit, int): 
        indices = topic.argsort().tolist()[:-limit - 1:-1]
    elif isinstance(limit, float): 
        indices = [i for i in topic.argsort().tolist()[::-1] if topic[i] >= limit]
    word_sims = [[features[i], topic[i]] for i in indices]
    return word_sims




# -------------- in-text span highlighting -------------------
def map_spans_into_doc(doc, spans):
    for start, end, label in spans:
        try:
            doc.ents += (Span(doc, start, end, label),)
        except:
            print('Issue with "{}" from pos. {} to {}'.format(doc[start:end].text, start, end))
    return doc



# taken from https://github.com/explosion/spacy-streamlit/blob/9592a27645f9bdb0c02c6add02838a506a0aaccf/spacy_streamlit/util.py#L26
def get_html(html: str):
    """Convert HTML so it can be rendered."""
    WRAPPER = """
    <div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; 
    padding: 1rem; margin-bottom: 2.5rem">{}</div>
    """
    # Newlines seem to mess with the rendering
    html = html.replace("\n", " ")
    return WRAPPER.format(html)



def get_spans_html(df_sents, df_spans, nlp, topic2color = None):
    html = ''
    sents = df_sents[['Para_id', 'Sent_id', 'text']].values.tolist()
    for para_id, sent_id, sent in sents:
        spans = df_spans[(df_spans.Para_id == para_id) & (df_spans.Sent_id == sent_id)][['start', 'end', 'topic_LSA']].values.tolist()
        doc = map_spans_into_doc(nlp(sent), spans)

        sent_html = displacy.render(
            doc, 
            style = "ent", 
            jupyter = False,
            options = ({'colors': topic2color} if topic2color else {}),
        )
        html += sent_html
    html = get_html(html)
    return html