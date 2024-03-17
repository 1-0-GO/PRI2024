from G11_code.helper_functions import *

'''
sentence_clustering(d,I,args)
    @input document d, optional inverted index I, optional clustering args

    @behavior identifies the best number of sentence clusters for the target tasks according
    to proper internal indices, returning the corresponding clustering solution

    @output clustering solution C
'''
def sentence_clustering(d, I, **args):
    pass

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