import numpy as np
import spacy
from collections import defaultdict
import nltk
from G11_code.data_collection import flatten
from G11_code.helper_functions import sort_by_value, tf_idf_term, log10_tf, select_and_sort
from G11_code.training import split
from G11_code.indexing import *
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from tqdm import trange
import warnings
import pandas as pd

def get_paragraph_features(article_file_path:str, sent_id:int):
    with open(article_file_path, "r", errors="ignore") as f:
        raw_text = f.read()
    paragraphs = raw_text.split("\n\n")[1:]
    sent_no = 0
    for paragraph_no in range(len(paragraphs)): 
        sents_par = nltk.sent_tokenize(paragraphs[paragraph_no])
        for in_par_sent_id in range(len(sents_par)):
            if sent_id == sent_no:
                return paragraph_no, in_par_sent_id
            sent_no +=1
    
def BM25_term(df_t, tf_t_d, N, s_len_avg, s_len, k, b): 
    idf_t = np.log10(N/df_t)
    B = 1 - b + b * (s_len/s_len_avg)
    return idf_t * (tf_t_d * (k + 1))/(tf_t_d + k * B)
            
def count_keyword_occurances(sentence:str, keywords:list):
    # Count occurrences of words with a preceding space or at the beginning of the sentence
    return sum(sentence.lower().count((' ' + word).lower()) for word in keywords)            

def get_scores_and_keywords(docs: list, sent_embeddings:list, doc_embeddings:list, I: list, k:0.2, b:0.75, p_keywords = 10 ):
    

    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    bert_scores, tfidf_scores, bm25_scores, keywords = dict(), dict(), dict(), dict()
    print("Bar to finish calculating scores and keywords")
    for d in tqdm(docs):
        
        
        doc_info = I.get_document_info(d)
        term_doc_info = zip(*doc_info.values())      
        num_terms_in_sentences = I.get_num_term_in_sentences(d)
        sentence_lengths = I.sentence_num_chars[d]
        num_sentences = len(num_terms_in_sentences)
        
        #TF-IDF Scores & TF-IDF Keywords        
        keyword_scores = dict()
        tfidf={key: 0 for key in range(num_sentences)}
        sq_normalization_term = defaultdict(int)
        for term, df_t, tf_t_d, tf_per_sentence in term_doc_info:
            rel_t_d = tf_idf_term(I.N, df_t, tf_t_d)
            keyword_scores[term] = rel_t_d
            for sent_number, tf_s_t, _ in tf_per_sentence:
                tfidf[sent_number] += rel_t_d * log10_tf(tf_s_t)
                sq_normalization_term[sent_number] = log10_tf(tf_s_t)**2
        # normalization
        for sent_number, score in tfidf.items():
            normalized_score = score / np.sqrt(sq_normalization_term[sent_number])
            tfidf[sent_number] =  normalized_score if np.isfinite(normalized_score) else 0
            
            
        tfidf = sort_by_value(tfidf, max_elements=len(tfidf), reverse=True)
        keywords[d] = sort_by_value(keyword_scores, max_elements=p_keywords, reverse=True)
        
        
        #We need to re-run this part. Because, apperently you can only loop once, for a zipped variable.
        doc_info = I.get_document_info(d)
        term_doc_info = zip(*doc_info.values())      
        num_terms_in_sentences = I.get_num_term_in_sentences(d)
        sentence_lengths = I.sentence_num_chars[d]
        num_sentences = len(num_terms_in_sentences)
        
        #BM-25 Scores
        bm25 = {key: 0 for key in range(num_sentences)}
        avg_sentence_length = sum(num_terms_in_sentences)/num_sentences
        for term, df_t, tf_t_d, tf_per_sentence in term_doc_info: 
            for sent_number, tf_s_t, _ in tf_per_sentence: 
                bm25[sent_number] += BM25_term(df_t, tf_s_t, I.N, avg_sentence_length, num_terms_in_sentences[sent_number], k, b)
        bm25 = sort_by_value(bm25, max_elements=len(bm25), reverse=True)
        
        #Bert Scores
        bert = dict()
        doc_embedding = doc_embeddings[d]
        for sent_id in range(num_sentences): 
            sent_vec = sent_embeddings[sent_id]
            score = np.squeeze(cosine_similarity(doc_embeddings[d], sent_embeddings[d][sent_id].reshape(1,-1)))
            bert[sent_id] = score.item()
        bert = sort_by_value(bert, max_elements=len(bert), reverse=True)
        
        bert_scores[d]=bert
        tfidf_scores[d]=tfidf
        bm25_scores[d] = bm25
        
    warnings.filterwarnings('default', category=RuntimeWarning)
    
    scores = {'bert': bert_scores, 'tfidf': tfidf_scores, 'bm25': bm25_scores}
    return scores, keywords

def get_named_entity_counts(docs:list, articles:list):
    nlp = spacy.load('en_core_web_sm')
    
    named_entity_counts = dict()
    print("Bar to finish calculating the named entity counts")
    for doc_id in tqdm(docs):        
        document = nltk.sent_tokenize(articles[doc_id])
        s_ent_count_dict = dict()
        for sent_id in range(len(document)):
            sentence = document[sent_id]
            s_ent_counts = {string: 0 for string in nlp.get_pipe("ner").labels}
            s_ent_labels = [ent.label_ for ent in nlp(sentence).ents]
            for string in s_ent_labels:
                s_ent_counts[string] += 1
            s_ent_counts = { (lambda k: k.lower() + '_count')(k) : v for k, v in s_ent_counts.items() }
            s_ent_count_dict[sent_id] = s_ent_counts
        named_entity_counts[doc_id] = s_ent_count_dict
            
    return named_entity_counts


def feature_extraction(s: int, d: int, **args):
    
    #Getting the required args values
    article_file_paths = ('article_file_paths' in args and args['article_file_paths']) or article_file_paths
    articles = ('articles' in args and args['articles']) or articles
    #The 'scores' in args has to be a tuple that contains all scores from BERT, TF-IDF, BM-25 in that order.
    scores = ('scores' in args and args['scores']) or None
    keywords = ('keywords' in args and args['keywords']) or None
    named_entity_counts = ('named_entity_counts' in args and args['named_entity_counts']) or None
    
    #Calculating the ranks
    ranks_bert = {key: rank for rank, key in enumerate(scores['bert'][d])} #The first one has to bert scores
    ranks_tfidf = {key: rank for rank, key in enumerate(scores['tfidf'][d])} #The second one has to tf-idf scores
    ranks_bm25 = {key: rank for rank, key in enumerate(scores['bm25'][d])} #The third one has to bm-25 scores
    
    #Getting the textual sentence to count keywords in the 
    sentence = nltk.sent_tokenize(articles[d])[s]
    named_entities = named_entity_counts[d][s]
    s_features = dict()
    s_features['document_id'] = d
    s_features['sent_id'] = s
    s_features['paragraph_no'], s_features['position_in_paragraph'] = get_paragraph_features(article_file_paths[d], s)
    s_features['score_bert'] = scores['bert'][d][s]
    s_features['score_tfidf'] = scores['tfidf'][d][s]
    s_features['score_bm25'] = scores['bm25'][d][s]
    s_features['rank_bert'] = ranks_bert[s]
    s_features['rank_tfidf'] = ranks_tfidf[s]
    s_features['rank_bm25'] = ranks_bm25[s]
    s_features['keyword_count'] = count_keyword_occurances(sentence, keywords=keywords[d].keys())
    s_features.update(named_entities)
  
    return s_features

def get_dataframe(docs:list, sent_embeddings:list, doc_embeddings:list,  I:list, article_file_paths:list, articles:list, k=0.2, b=0.75, p_keywords=10):
    
    scores, keywords = get_scores_and_keywords(docs=docs, sent_embeddings=sent_embeddings, doc_embeddings= doc_embeddings, I=I, k=k, b=b, p_keywords=p_keywords)
    data = []
    named_entity_counts = get_named_entity_counts(docs=docs, articles=articles)
    print("Bar to finish gathering all features for given documents")
    for doc_id in tqdm(docs):
        for sent_id in range(len(I.sentence_num_chars[doc_id])):
            s_features = feature_extraction(s=sent_id, d=doc_id, 
                               article_file_paths=article_file_paths,
                               articles = articles,
                               scores = scores,
                               keywords = keywords,
                               named_entity_counts = named_entity_counts)
            data.append(s_features)
    return data



def construct_df_and_split(doc_ids_by_cat:list, summary_sentence_indices_by_cat:list, sent_embeddings:list, doc_embeddings:list, article_file_paths:list, articles:list, train_size=0.8, k=0.2, b=0.75, p_keywords=10):
    category_group = zip(doc_ids_by_cat, summary_sentence_indices_by_cat)
    
    X_train_ids, y_train_sent_ids, X_test_ids, y_test_sent_ids = list(), list(), list(), list()
    
    for doc_ids_by_cat, sentence_indices in category_group: 
        train_id, test_id, train_sums, test_sums = split(X=doc_ids_by_cat, Y=sentence_indices, train_ratio=train_size)
        
        train_doc_sent_indice_pairs = zip(train_id, train_sums)
        train_sum_sent_ids = list()
        for train_doc_id, train_sum_sent_indices in train_doc_sent_indice_pairs:
            for sum_sent_ids in train_sum_sent_indices:
                train_sum_sent_ids.append([train_doc_id, sum_sent_ids])
                
        test_doc_sent_indice_pairs = zip(test_id, test_sums)
        test_sum_sent_ids = list()
        for test_doc_id, test_sum_sent_indices in test_doc_sent_indice_pairs:
            for sum_sent_ids in test_sum_sent_indices:
                test_sum_sent_ids.append([test_doc_id, sum_sent_ids])
        
        X_train_ids.append(train_id)
        y_train_sent_ids.append(train_sum_sent_ids)
        X_test_ids.append(test_id)
        y_test_sent_ids.append(test_sum_sent_ids)
        
    X_train_ids = flatten(X_train_ids)
    y_train_sent_ids = flatten(y_train_sent_ids)
    X_test_ids = flatten(X_test_ids)
    y_test_sent_ids = flatten(y_test_sent_ids)
    
    train_index = indexing(articles)
    train_index.recalculate_dfs(X_train_ids)
    
    test_index = indexing(articles)
    test_index.recalculate_dfs(X_test_ids)

    X_train = get_dataframe(docs=X_train_ids, sent_embeddings=sent_embeddings, doc_embeddings=doc_embeddings,  I=train_index, article_file_paths=article_file_paths, articles=articles, k=k, b=b, p_keywords=p_keywords)
    X_train = pd.DataFrame(X_train)
    y_train = list()
    for index, x_train_row in X_train.iterrows():
        if [x_train_row['document_id'],x_train_row['sent_id']] in y_train_sent_ids:
            y_train.append(1)
        else:
            y_train.append(0)
        
    X_test = get_dataframe(docs=X_test_ids, sent_embeddings=sent_embeddings, doc_embeddings=doc_embeddings,  I=test_index, article_file_paths=article_file_paths, articles=articles, k=k, b=b, p_keywords=p_keywords)
    X_test = pd.DataFrame(X_test)
    y_test = list()
    for index, x_test_row in X_test.iterrows():
        if [x_test_row['document_id'],x_test_row['sent_id']] in y_test_sent_ids:
            y_test.append(1)
        else:
            y_test.append(0)
    return X_train, y_train, X_test, y_test


def supervised_summarization(d:int, M, p=7, l=0, **args):
    
    o= ('o' in args and args['o']) or "rel"
    
    if "new_document" in args:
        test_index = ('test_index' in args and args['test_index']) or test_index
        sent_embeddings = ('sent_embeddings' in args and args['sent_embeddings']) or sent_embeddings
        doc_embeddings = ('doc_embeddings' in args and args['doc_embeddings']) or doc_embeddings
        article_file_paths = ('article_file_paths' in args and args['article_file_paths']) or article_file_paths
        articles = ('articles' in args and args['articles']) or articles
        k = ('k' in args and args['k']) or 0.2
        b = ('b' in args and args['b']) or 0.75
        p_keywords = ('p_keywords' in args and args['p_keywords']) or 10
        frame = get_dataframe(docs=[d], sent_embeddings=sent_embeddings, doc_embeddings=doc_embeddings,  I=test_index, article_file_paths=article_file_paths, articles=articles, k=k, b=b, p_keywords=p_keywords)
        frame = pd.DataFrame(frame)
    else:
        x_test = ('x_test' in args and args['x_test']) or x_test
        frame = x_test[x_test['document_id']==d]
        test_index = ('test_index' in args and args['test_index']) or test_index
    
    sentence_lengths = test_index.sentence_num_chars[d]
    
    scores = {sent_id: M.predict(frame[frame['sent_id']==sent_id]) for sent_id in frame['sent_id']}
    
    return select_and_sort(scores=scores, o=o, p=p, l=l, sentence_lengths=sentence_lengths)