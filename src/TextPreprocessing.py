
import sys
import os

# for text
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from sklearn.feature_extraction.text import TfidfVectorizer




def get_ngrams(texts, ngram_range):
    vectorizer = TfidfVectorizer(
        use_idf = True,
        ngram_range = (1, ngram_range),
        min_df = 1,
        lowercase = True,
        stop_words = [],
        strip_accents = None,
    ).fit(texts)
    
    ngrams = vectorizer.get_feature_names()
    return ngrams



def strip_ngrams_fr(ngrams) :
    ngrams_stripped = []
    ngrams.sort(key = len, reverse = True)
    for ngram in ngrams :
        cut = ngram.split()
        somehow_goods = [i for i, w in enumerate(cut) 
                 if w.lower() not in list(fr_stop) + ['le', 'la', 'l', 'les', 'de', 
                              'du', 'des', 'd', 's', 'se', 'ce', 'ces', 'cette', 
                              'un', 'une', 'et', 'avec']]
        goods = [i for i, w in enumerate(cut) 
                 if w.lower() not in list(fr_stop) + ['le', 'la', 'l', 'les', 'de', 
                              'du', 'des', 'd', 'en', 'est', 's', 'se',
                              'un', 'une', 'et', 'sur', 'pour', 
                              'avec', 'par', 'dans', 'a', 'à',  'au', 'aux']]
        if goods != [] : 
            ngram_stripped = ' '.join(cut[min(somehow_goods):max(goods)+1])
            ngrams_stripped.append(ngram_stripped)
            
    ngrams_stripped = list(set(ngrams_stripped))
    return ngrams_stripped
