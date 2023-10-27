# for data
import scipy
import numpy as np

# for viz
import seaborn as sns
import matplotlib.pyplot as plt

# for graph
from pyvis import network as nt





# -------------- similarity utils -------------------
def get_most_similar_indices(sim_matrix, n, idx):
    ids = np.argsort(-sim_matrix[idx].toarray())[0].tolist()[:n]
    return [(i, sim_matrix[idx, i]) for i in ids if sim_matrix[idx, i] > 0]



def filter_similarity_matrix(
    df_corpus, 
    sim_matrix, 
    key_column, 
    x_key = None, 
    y_key = None, 
    idx = None,
    ):
    x_ids = None
    y_ids = None
    if idx is not None:
        x_key = df_corpus.at[idx, key_column]
        x_ids = df_corpus[df_corpus[key_column] == x_key].index.tolist()
        y_key = x_key
        y_ids = x_ids
        sim_matrix = sim_matrix[x_ids, :][:, y_ids]
    else:
        # filter sim matrix along rows
        if x_key:
            x_ids = df_corpus[df_corpus[key_column] == x_key].index.tolist()
            sim_matrix = sim_matrix[x_ids, :]

        # filter sim matrix along columns
        y_key = (y_key if y_key else x_key)
        if y_key:
            y_ids = df_corpus[df_corpus[key_column] == y_key].index.tolist()
            sim_matrix = sim_matrix[:, y_ids]
    return (sim_matrix, x_ids, y_ids)





# -------------- similarity graph -------------------
def build_nt_graph(sim_matrix, height = 1200, width = 1980, notebook = False):
    cx = scipy.sparse.coo_matrix(sim_matrix)
    
    nodes = list(range(sim_matrix.shape[0]))
    edges = [(int(i), int(j), float(v)) for i, j, v in zip(cx.row, cx.col, cx.data)]

    nt_graph = nt.Network(directed = False, height = height, width = width, notebook = notebook)
    nt_graph.add_nodes(
        nodes = nodes,
        label = [str(i) for i in nodes],
        title = [str(i) for i in nodes],
    )
    nt_graph.add_edges(edges)
    return nt_graph




# -------------- similarity plotting -------------------
def get_similarity_heatmap(
    sim_matrix, 
    x_labels = None, 
    y_labels = None, 
    incident_idx = None, 
    legend = '',
    ):
    '''Returns a triangular heatmap of correlations between incidents'''
    x_labels = (x_labels if x_labels else list(range(1, sim_matrix.shape[1] + 1)))
    y_labels = (y_labels if y_labels else list(range(1, sim_matrix.shape[0] + 1))) 
    
    if incident_idx is not None:
        sim_matrix = sim_matrix[[y_labels.index(incident_idx)], :]
        y_labels = [incident_idx]

    xshape = min(20, int(sim_matrix.shape[1]/2)+1)
    yshape = max(5,  int(sim_matrix.shape[0]/2)+1)
    
    fig = plt.figure(figsize = (xshape, yshape))
    cmap = sns.diverging_palette(220, 10, as_cmap = True) #"BuPu"
    hmap = sns.heatmap(
        sim_matrix, 
        vmin = 0., 
        vmax = 1.,
        xticklabels = x_labels,
        yticklabels = y_labels,
        cmap = cmap, 
        square = True,
        linewidth = .15, 
        cbar_kws = {"shrink": .5},
    )
    # hmap.invert_yaxis()
    hmap.xaxis.tick_top()
    if legend: 
        hmap.set_title(legend + '\n\n')
    return fig