
import sys
import os

# for data
import pandas as pd
import numpy as np

# for text
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from sklearn.feature_extraction.text import TfidfVectorizer

# for clustering
from sklearn.decomposition import PCA, NMF, LatentDirichletAllocation, TruncatedSVD




class unsupervizedEntityClassifier():
    def __init__(self, vectorizer, model, vocab = None, method = 'LSA', topic_dict = None):
        self.vectorizer = vectorizer
        self.model = model
        self.vocab = vocab
        self.method = method
        self.components = model.explained_variance_.shape[0]
        topic_dict = {i: 'Entity {}'.format(i) for i in range(1, self.components+1)} if not topic_dict else topic_dict
        self.set_topics(topic_dict)

    def set_topics(self, topic_dict):
        self.topic_dict = topic_dict
        self.idx2key = {i: k for i, k in enumerate(self.topic_dict.keys())}
        return

    def explain_topics(self, n_top_words = 15):
        features = self.vectorizer.get_feature_names()
        if self.vocab:
            features = [w for w in features if w in self.vocab]

        # (topic * feature) matrix
        df_topic_feature = pd.DataFrame(np.round(self.model.components_, 4))
        df_topic_feature.columns = features
    
        n_row_plots = int((self.components-0.1)/3)+1
        fig, axes = plt.subplots(n_row_plots, 3, figsize = (25, 8*n_row_plots), sharex = True)
        axes = axes.flatten()
        for topic_idx, topic in enumerate(df_topic_feature.values):
            top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
            top_features = [features[i] for i in top_features_ind]
            weights = topic[top_features_ind]

            ax = axes[topic_idx]
            ax.barh(top_features, weights, height = 0.7)
            ax.set_title(f'Topic {topic_idx+1}', fontdict = {'fontsize': 30})
            ax.invert_yaxis()
            ax.tick_params(axis = 'both', which = 'major', labelsize = 20)
            for i in 'top right left'.split():
                ax.spines[i].set_visible(False)

        plt.subplots_adjust(top = 0.90, bottom = 0.05, wspace = 0.90, hspace = 0.3)
        plt.show()
        return
        
    def vectorize(self, X):
        vects = self.vectorizer.transform(X)
        feats = self.vectorizer.get_feature_names()
        if self.vocab:
            feats = [i for i, w in enumerate(feats) if w in self.vocab]
            vects = vects[:, feats]
        return vects
    
    def predict(self, X):
        X = self.vectorize(X)
        y = self.model.transform(X)
        y = y[:, [k-1 for k in self.topic_dict.keys()]]
        probas = y.max(axis = 1)
        labels = y.argmax(axis = 1)
        labels = [self.topic_dict[self.idx2key[l]] for l in labels]
        output = [(l, p) for l, p in zip(labels, probas)]
        return output



def fit_unsupervized_entity_classifier(
    texts, 
    method = 'LSA', 
    n_components = None, 
    vocab = None,
    stop_words = [],
):
    if method not in ['PCA', 'LSA', 'LDA']:
        print('Invalid method (choose between PCA, LSA and LDA)')
        return

    # fit (text * feature) matrix
    vectorizer = TfidfVectorizer(
        sublinear_tf = False,
        ngram_range = (1, 3),
        min_df = 3,
        stop_words = stop_words,
        strip_accents = None,
        lowercase = True,
    )
    vects = vectorizer.fit_transform(texts) # size (num_texts, num_features)
    feats = vectorizer.get_feature_names()
    
    # reduce feature size
    if vocab:
        feats = [i for i, w in enumerate(feats) if w in vocab]
        vects = vects[:, feats]

    # fit (text * topic) matrix
    if method == 'PCA' : 
        model = PCA(
            n_components = n_components, 
            random_state = 42)
        model = model.fit(vects.todense())
        
    elif method == 'LSA' : 
        model = TruncatedSVD(
            n_components = (n_components if n_components else min(vects.shape)-1), 
            random_state = 42)
        model = model.fit(vects)
        
    elif method == 'LDA' : 
        model = LatentDirichletAllocation(
            n_components = (n_components if n_components else min(vects.shape)-1), 
            random_state = 42)
        model = model.fit(vects)
    
    # ship into pipeline
    pipeline = unsupervizedEntityClassifier(vectorizer, model, vocab, method)
    return pipeline



def load_entity_classifier(path_to_file):
    with open(path_to_file, 'rb') as file: 
        entity_classifier = dill.load(file)
    return entity_classifier