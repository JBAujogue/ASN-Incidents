
import sys
import os
import copy

# for data
import pandas as pd
import numpy as np
from scipy.sparse import hstack, vstack

# for text similarity
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer

# for viz
import matplotlib.pyplot as plt
import seaborn as sns


def get_tfidf(texts, min_df = 1) :
    '''Returns the ticket-ticket correlation matrix for each machine,
       sorted by time and with lower tringular values only.
    '''
    tfidf = TfidfVectorizer(
        sublinear_tf = True,
        ngram_range = (1, 3),
        stop_words = [],
        min_df = min_df,
        lowercase = True,
    ).fit(texts)
    return tfidf


def get_correlations_by_key(key2tfidf) :
    '''
    Returns the ticket-ticket correlation matrix for each machine,
    sorted by time and with lower tringular values only.
    '''
    key2corrs = {}
    
    for key1, vects1 in key2tfidf.items():
        key2corrs[key1] = {}
        vects1 = normalize(vects1)     # size (num_texts1, num_features)
        for key2, vects2 in key2tfidf.items():
            vects2 = normalize(vects2) # size (num_texts2, num_features)            
            corrs = vects1 * vects2.T  # size (num_texts1, num_texts2)
            corrs = corrs.toarray()
            #corrs = np.tril(corrs, k = -1)
            #corrs = np.flip(corrs, axis = 0)
            key2corrs[key1][key2] = corrs
    return key2corrs


def get_correlations(key2tfidf) :
    '''
    Returns the ticket-ticket correlation matrix for each machine,
    sorted by time and with lower tringular values only.
    '''
    vects = vstack(key2tfidf.values())
    vects = normalize(vects)           # size (num_texts, num_features)
    corrs = vects * vects.T            # size (num_texts, num_texts)
    corrs = corrs.toarray()
    return corrs


def get_correlations_plot(corrs, x_labels = None, y_labels = None, incident_idx = None, legend = '') :
    '''Returns a triangular heatmap of correlations between incidents'''
    
    vmin = np.min(corrs)
    vmax = np.max(corrs)
    
    if incident_idx is not None :
        x_labels = x_labels if x_labels else list(range(1, corrs.shape[1] + 1))
        y_labels = [incident_idx]
        corrs = corrs[[incident_idx], :]
    else :
        x_labels = x_labels if x_labels else list(range(1, corrs.shape[1] + 1))
        y_labels = y_labels if y_labels else list(range(1, corrs.shape[0] + 1)) 

    xshape = int(corrs.shape[1]/2)+1
    yshape = max(5, int(corrs.shape[0]/2)+1)
    
    fig = plt.figure(figsize = (xshape, yshape))
    cmap = sns.diverging_palette(220, 10, as_cmap = True) #"BuPu"
    hm = sns.heatmap(
        corrs, 
        vmin = vmin, 
        vmax = vmax,
        xticklabels = x_labels,
        yticklabels = y_labels,
        cmap = cmap, 
        square = True,
        linewidth = .15, 
        cbar_kws = {"shrink": .5},
    )
    hm.invert_yaxis()
    hm.xaxis.tick_top()
    if legend : hm.set_title(legend + '\n\n')
    return fig



class Corrélateur(object) :
    def __init__(self, df, text_column, key_column, min_df = 1, entities = None):
        
        # get clean texts
        self.df = df
        self.text_column = text_column
        self.key_column = key_column
        self.keys = df[key_column].unique().tolist()
        self.key2df = {k: df[df[key_column] == k] for k in self.keys}
        self.key2texts = {k: v[text_column].tolist() for k, v in self.key2df.items()}
        
        # similarities computed on plain tickets
        self.vectorizer = get_tfidf(df[text_column].tolist(), min_df = min_df)
        self.key2tfidf = {k: self.vectorizer.transform(v) for k, v in self.key2texts.items()}
        self.key2corrs = get_correlations_by_key(self.key2tfidf)
        self.corrs = get_correlations(self.key2tfidf)
        
    def display_text(self, key, incident_num):
        ticket = self.key2texts[key][incident_num-1]
        print('Incident n° ' + str(incident_num))
        print(ticket)
        return

    def get_correlations_plot(self, key1 = None, key2 = None, incident_num = None):
        if incident_num:
            incident_idx = incident_num-1
            key1 = self.df[self.key_column][incident_idx]
            relative_idx = self.key2df[key1].index.tolist().index(incident_idx)
        else:
            relative_idx = None
        
        # inter-key similarities
        if (key1 and key2):
            corrs = self.key2corrs[key1][key2]
            x_labels = [i+1 for i in self.key2df[key2].index.tolist()]
            y_labels = [i+1 for i in self.key2df[key1].index.tolist()]
            legend = (
                (key1 + ' : incident n° ' + str(incident_num) + '\n versus \n ' + key2) 
                if incident_num else key1 + ' (vertical)\n versus \n' + key2 + ' (horizontal)'
            )
            hm = get_correlations_plot(corrs, x_labels, y_labels, relative_idx, legend)
        
        # intra-key similarities
        elif (key1 or key2):
            key = (key1 if key1 else key2)
            corrs = self.key2corrs[key][key]
            x_labels = [i+1 for i in self.key2df[key].index.tolist()]
            y_labels = [i+1 for i in self.key2df[key].index.tolist()]
            legend = (
                (key + ' : incident n° ' + str(incident_num)) 
                if incident_num else key
            )
            hm = get_correlations_plot(corrs, x_labels, y_labels, relative_idx, legend)
        else:
            print('Invalid input')
        return hm

    def get_most_similar(self, incident_num, key2 = None, topk = 1):
        incident_idx = incident_num-1
        key1 = self.df[self.key_column][incident_idx]
        key2 = key2 if key2 else key1

        key1_idxs = self.key2df[key1].index.tolist()
        key2_idxs = self.key2df[key2].index.tolist()
        relative_idx = key1_idxs.index(incident_idx)

        corrs = self.key2corrs[key1][key2]

        # retrieve top most similar tickets
        idxs = np.flip(np.argsort(corrs[relative_idx, :]))[1:topk+1].tolist()
        texts = [self.key2texts[key2][i] for i in idxs]
        sims = corrs[relative_idx, idxs].tolist()
        idxs = [key2_idxs[i]+1 for i in idxs]
            
        return (idxs, texts, sims)
