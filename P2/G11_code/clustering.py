from G11_code.helper_functions import *
from G11_code.data_collection import *
from G11_code.indexing import InvertedIndex
import numpy as np
import nltk
import itertools
from sklearn.metrics.pairwise import cosine_distances
import kmedoids
from sklearn.cluster import AgglomerativeClustering


def dunn_index(dissimilarity_matrix: np.array, clusters: dict, inter_cluster='average', intra_cluster='complete'):
    def get_distances_between(c_i, c_j):
        rows, cols = zip(*itertools.product(clusters[c_i], clusters[c_j]))
        return dissimilarity_matrix[rows, cols]
    def inter_cluster_dist_single(c_i, c_j):
        return np.min(get_distances_between(c_i, c_j))       
    def inter_cluster_dist_complete(c_i, c_j):
        return np.max(get_distances_between(c_i, c_j))       
    def inter_cluster_dist_average(c_i, c_j):
        return np.mean(get_distances_between(c_i, c_j))       
    def intra_cluster_dist_single(c_i):
        rows = clusters[c_i]
        return np.min(dissimilarity_matrix[rows, rows])
    def intra_cluster_dist_complete(c_i):
        rows = clusters[c_i]
        return np.max(dissimilarity_matrix[rows, rows])
    def intra_cluster_dist_average(c_i):
        rows = clusters[c_i]
        return np.mean(np.triu(dissimilarity_matrix[rows, rows]))
    get_inter_cluster = {'single': inter_cluster_dist_single, 'complete': inter_cluster_dist_complete, 'average': inter_cluster_dist_average}
    get_intra_cluster = {'single': intra_cluster_dist_single, 'complete': intra_cluster_dist_complete, 'average':intra_cluster_dist_average} 
    inter_cluster_function = get_inter_cluster[inter_cluster]
    intra_cluster_function = get_intra_cluster[intra_cluster]
    cohesion = float('inf')
    for c_i in range(len(clusters)):
        c_j = c_i + 1
        while c_j < len(clusters):
            cohesion = min(cohesion, inter_cluster_function(c_i, c_j))
    separation = max([intra_cluster_function(c_i) for c_i in range(len(clusters))])
    return cohesion / separation
                
def lower_diagonal_mask(a: np.array, k: int=0) -> np.array:
    mask = np.tril(np.ones_like(a), k=k)
    return np.ma.masked_where(mask, a)

def tf_idf_compute_dissimilarity_matrix(d: int, I: InvertedIndex, conversion_function=lambda S: 1-S):
    doc_info = I.get_document_info(d)
    term_doc_info = list(zip(*doc_info.values()))
    num_terms_in_sentences = I.get_num_term_in_sentences(d)
    num_sentences = len(num_terms_in_sentences)
    similarity_matrix = compute_similarities_between_sentences(term_doc_info, I.N, num_sentences)
    similarity_matrix = np.clip(similarity_matrix, 0, 1)
    dissimilarity_matrix = conversion_function(similarity_matrix)
    return lower_diagonal_mask(dissimilarity_matrix)

def bert_compute_dissimilarity_matrix(d: int, D: list=[], bert_params: tuple=(), file_path: str=""):
    if file_path != "" and os.path.isfile(file_path):
        embeddings = flatten(pickle_load(file_path))[d]
    else:
        try:   
            tokenizer, model, device = bert_params
        except ValueError:
            raise Exception("Sorry, but you need to provide either a file_path with the bert embeddings or the bert_params tuple with (tokenizer, model, device).")
        try: 
            document = D[d]
        except IndexError:
            if D:
                raise Exception(f"There's only {len(D)} documents in the document collection (D) you provided, but you are asking for the summary of document number d={d}.")
            raise Exception("Are you sure you provided a document collection (D=)?")
        sentences = nltk.sent_tokenize(document)
        embeddings = get_embeddings(sentences, tokenizer, model, device)
    dissimilarity_matrix = cosine_distances(embeddings)
    dissimilarity_matrix = np.clip(dissimilarity_matrix, 0, 1)
    return lower_diagonal_mask(dissimilarity_matrix)

'''
sentence_clustering(dissimilarity_matrix,args)
    @input a dissimilarity matrix for a given document, optional clustering args

    @behavior identifies the best number of sentence clusters for the target tasks according
    to proper internal indices, returning the corresponding clustering solution

    @output num_clusters, clustering solution C
'''
def sentence_clustering(dissimilarity_matrix, algorithm='k-medoids', **args):
    kmax = ('kmax' in args and args['kmax']) or len(dissimilarity_matrix)
    match algorithm: 
        case 'k-medoids':
            clustering_model = kmedoids.KMedoids(n_clusters = kmax, 
                                         method = 'dynmsc',
                                         max_iter=5000)
            clustering_model.fit(dissimilarity_matrix)
        case 'agglomerative':
            clustering_model = AgglomerativeClustering(linkage='average')
    return len(set(clustering_model.labels_)), clustering_model
    

'''
summarization(d,C,I,args)
    @input document d, sentence clusters C, optional inverted index I and guiding args

    @behavior ranks non-redundant sentences from the clustering solution, taking into
    attention that ranking criteria can be derived from the clusters' properties

    @output summary (without pre-fixed size limits)
'''
def summarization(d, C, I, **args):
    pass

'''
keyword_extraction(d,C,I,args)
    @input document d, sentence clusters C, optional inverted index I and guiding args

    @behavior extracts non-redundant keywords from the clustering solution, taking into
    consideration that multiple clusters may share high-relevant terms

    @output set of keywords (without pre-fixed cardinality)
'''
def keyword_extraction(d, C, I, **args):
    pass