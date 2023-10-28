# for text
import nltk
from spacy.lang.en.stop_words import STOP_WORDS as en_stop
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from gensim.corpora import Dictionary
from gensim.similarities import SparseMatrixSimilarity
from gensim.models import TfidfModel
from sklearn.feature_extraction.text import TfidfVectorizer






# -------------- gensim tfidf utils -------------------
def compute_gensim_tfidf_matrix(corpus, tokenizer = nltk.word_tokenize):
    corpus_tokenized = [tokenizer(s) for s in corpus]
    dictionary = Dictionary(corpus_tokenized)
    corpus_bow = [dictionary.doc2bow(t) for t in corpus_tokenized]
    tfidf_model = TfidfModel(corpus_bow)
    corpus_tfidf = tfidf_model[corpus_bow]
    return (corpus_tfidf, dictionary)



def compute_gensim_tfidf_similarity_matrix(
    corpus, 
    tokenizer = nltk.word_tokenize, 
    num_best = 500, 
    cancel_diagonal = True,
    threshold = 0.25,
    ):
    corpus_tfidf, dictionary = compute_gensim_tfidf_matrix(corpus, tokenizer = tokenizer)

    sim_model = SparseMatrixSimilarity(
        corpus = corpus_tfidf, 
        num_docs = len(corpus_tfidf), 
        num_terms = len(dictionary),
        num_best = num_best,
        maintain_sparsity = True,
    )
    # compute sparse csr matrix
    sim_matrix = sim_model[corpus_tfidf]
    if cancel_diagonal:
        sim_matrix.setdiag(0)
        sim_matrix.eliminate_zeros()
    if threshold > 0:
        sim_matrix = sim_matrix.multiply(sim_matrix > threshold)
        
    # clip to 1.0
    sim_matrix = sim_matrix.multiply(sim_matrix <= 1.) + 1.*(sim_matrix > 1.)
    return sim_matrix





# -------------- sklearn tfidf utils -------------------
def compute_stripped_ngrams_fr(corpus, *args, **kwargs):
    # compute ngrams
    vectorizer = TfidfVectorizer(*args, **kwargs).fit(corpus)
    ngrams = vectorizer.get_feature_names_out()
    ngrams = sorted(ngrams, key = len, reverse = True)
    
    # strip ngrams
    ngrams_stripped = []
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



def compute_sklearn_tfidf_matrix(corpus, vocab = None, *args, **kwargs):
    vectorizer = TfidfVectorizer(*args, **kwargs)
    
    # (text * feature) matrix
    tfidf_matrix = vectorizer.fit_transform(corpus)
    tfidf_ngrams = vectorizer.get_feature_names_out()
    
    # reduce feature size
    if vocab:
        inds = [i for i, w in enumerate(tfidf_ngrams) if w in vocab]
        tfidf_ngrams = [w for w in tfidf_ngrams if w in vocab]
        tfidf_matrix = tfidf_matrix[:, inds]
    return (tfidf_matrix, tfidf_ngrams)






