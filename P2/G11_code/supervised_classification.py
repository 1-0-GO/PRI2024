import numpy as np
import spacy
from collections import defaultdict
import nltk
from G11_code.helper_functions import sort_by_value, tf_idf_term, log10_tf
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from tqdm import trange
import warnings


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
            for sent_number, tf_s_t in tf_per_sentence:
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
            for sent_number, tf_s_t in tf_per_sentence: 
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