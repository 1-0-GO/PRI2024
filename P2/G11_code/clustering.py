from G11_code.helper_functions import *
from G11_code.data_collection import *
from G11_code.indexing import InvertedIndex
from collections import defaultdict
import numpy as np
import nltk
import kmedoids
from sklearn.metrics.pairwise import cosine_distances
import scipy.cluster.hierarchy as sch
from sklearn.metrics import silhouette_score

def transform_labels(labels):
    n_clusters = len(set(labels))
    clusters =  [[] for _ in range(n_clusters)]
    min_ci = min(labels)
    for i, label in enumerate(labels):
        clusters[label - min_ci].append(i)
    return n_clusters, clusters

def dunn_index(dissimilarity_matrix: np.array, labels: np.array, inter_cluster='average', intra_cluster='complete'):
    n_clusters, clusters = transform_labels(labels)
    def get_distances_between(c_i, c_j):
        clust_c_i = clusters[c_i]
        clust_c_j = clusters[c_j]
        rows = np.repeat(clust_c_i, len(clust_c_j))
        cols = np.tile(clust_c_j, len(clust_c_i))
        return dissimilarity_matrix[rows, cols]
    def inter_cluster_dist_single(c_i, c_j):
        return np.min(get_distances_between(c_i, c_j))       
    def inter_cluster_dist_complete(c_i, c_j):
        return np.max(get_distances_between(c_i, c_j))       
    def inter_cluster_dist_average(c_i, c_j):
        return np.mean(get_distances_between(c_i, c_j))       
    def intra_cluster_dist_single(c_i):
        return (len(clusters[c_i])>1 and np.min(get_distances_between(c_i, c_i))) or 0
    def intra_cluster_dist_complete(c_i):
        return (len(clusters[c_i])>1 and np.max(get_distances_between(c_i, c_i))) or 0
    def intra_cluster_dist_average(c_i):
        return (len(clusters[c_i])>1 and np.mean(get_distances_between(c_i, c_i))) or 0
    get_inter_cluster_option = {'single': inter_cluster_dist_single, 'complete': inter_cluster_dist_complete, 'average': inter_cluster_dist_average}
    get_intra_cluster_option = {'single': intra_cluster_dist_single, 'complete': intra_cluster_dist_complete, 'average':intra_cluster_dist_average} 
    inter_cluster_function = get_inter_cluster_option[inter_cluster]
    intra_cluster_function = get_intra_cluster_option[intra_cluster]
    separation = float('inf')
    for c_i in range(n_clusters):
        for c_j in range(c_i + 1, n_clusters):
            separation = min(separation, inter_cluster_function(c_i, c_j))
    cohesion = max([intra_cluster_function(c_i) for c_i in range(n_clusters)]) or float('inf')
    return separation / cohesion
                
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
sentence_clustering(d, metric, algorithm, args)
    @input 
    d: array_like or int
        A dissimilarity matrix for a given document if metric=='precomputed', document id otherwise.

    metric: str or function
        'precomputed' by default. Can be function that takes a document id and returns a dissimilarity matrix (e.g. f = lambda d: tf_idf_compute_dissimilarity_matrix(d, I)).

    algorithm: str
        Algorithm used to compute the clustering. One of 'k-medoids' or 'agglomerative' By default it's k-medoids.
    
    optional clustering args:
        evaluate: str
            If algorithm='agglomerative' then paramater 'evaluate' can be used to specify the evaluation metric optimized when cutting the dendogram. By default, it's the silhouette_score. 'evaluate' should be a function that receives a dissimilarity matrix and labels and outputs the metric's value.
        linkage: str
            If algorithm='agglomerative' then parameter linkage can be used to specify the linkage method. Default is 'average'.
        kmax: int
            Maximum number of clusters.
        
    @behavior 
    identifies the best number of sentence clusters for the target tasks according
    to proper internal indices, returning the corresponding clustering solution

    @output 
    num_clusters, clustering solution C (C[i] is the cluster number to which sentence i belongs).
'''
def sentence_clustering(d, metric='precomputed', algorithm='k-medoids', **args):
    if metric == 'precomputed':
        dissimilarity_matrix = d
    else:
        dissimilarity_matrix = metric(d)
    kmax = ('kmax' in args and args['kmax']) or len(dissimilarity_matrix)
    match algorithm: 
        case 'k-medoids':
            clustering_model = kmedoids.KMedoids(n_clusters = kmax, 
                                         method = 'dynmsc',
                                         max_iter=5000)
            clustering_model.fit(dissimilarity_matrix)
            labels = clustering_model.labels_
            return len(set(labels)), (labels, clustering_model.medoid_indices_)
        case 'agglomerative':
            evaluate = ('evaluate' in args and args['evaluate']) \
                or (lambda dM,labs: silhouette_score(dM, labs, metric='precomputed'))
            linkage = ('linkage' in args and args['linkage']) or 'average'
            Z = sch.linkage(dissimilarity_matrix.compressed(), method=linkage)
            kmax += 1
            max_score = np.NINF
            max_labels = None
            for k in range(2, kmax):
                labels = sch.fcluster(Z, k, criterion='maxclust')
                score = evaluate(dissimilarity_matrix, labels)
                if score > max_score:
                    max_labels = labels
                    max_score = score
            labels = max_labels
    return len(set(labels)), labels
    

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