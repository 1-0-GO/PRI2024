import numpy as np
from collections import defaultdict
import torch
import json
import nltk
import pickle
import os
from tqdm import tqdm

def log10_tf(x):
    try:
        return (1 + np.log10(x)) 
    except ValueError:
        return 0
    
def tf_idf_term(N, df_t, tf_t_d):
    df_t = df_t or 1
    return log10_tf(tf_t_d) * np.log10(N/df_t)

def tf_idf_relevance_scores(info: list, N: int):
    scores = defaultdict(int)
    sq_normalization_term = defaultdict(int)
    for _, df_t, tf_t_d, tf_per_sentence in info:
        rel_t_d = tf_idf_term(N, df_t, tf_t_d)
        for sent_number, tf_s_t in tf_per_sentence:
            scores[sent_number] += rel_t_d * log10_tf(tf_s_t)
            sq_normalization_term[sent_number] = log10_tf(tf_s_t)**2
    # normalization
    for sent_number, score in scores.items():
        scores[sent_number] = score / np.sqrt(sq_normalization_term[sent_number])
    return scores

def compute_similarities_between_sentences(info: list, N: int, num_sentences: int):
    similarity_matrix = np.zeros((num_sentences, num_sentences))
    sq_norm_sentence = np.zeros(num_sentences)
    for _, df_t, tf_t_d, tf_per_sentence in info:
        log10_tfs = np.zeros(num_sentences)
        for sent_number, tf_s_t in tf_per_sentence:
            log10_tfs[sent_number] = log10_tf(tf_s_t)
        idf_term = np.log10(N/df_t)
        scores_for_term = idf_term * np.outer(log10_tfs, log10_tfs)
        similarity_matrix += scores_for_term
        sq_norm_sentence += idf_term * log10_tfs**2
    norm_sentence = np.sqrt(sq_norm_sentence)
    normalization = np.outer(norm_sentence, norm_sentence)
    res = np.divide(similarity_matrix, normalization, out=np.zeros_like(similarity_matrix), where=normalization!=0)
    return res

def sort_by_value(d: dict, max_elements: int, reverse=False) -> dict: 
    return dict(sorted(d.items(), key=lambda item: item[1], reverse=reverse)[:max_elements])

def sort_by_key(d: dict) -> dict: 
    return dict(sorted(d.items()))

def get_embeddings(sentences: list, tokenizer, model, device) -> np.array: 
    # runs much faster on gpu (10x)
    encoded_input = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**encoded_input)
    features = (outputs['last_hidden_state'][:,0,:]).cpu().numpy()
    return features

def summary_compute(categorized_set_of_articles, summarization_function, path_to_articleID):
    all_scores = list()
    for category_id, category in enumerate(categorized_set_of_articles): 
        all_scores.append([])
        for cat_path in category: 
            article_id = path_to_articleID(cat_path)
            article_score = summarization_function(article_id)
            all_scores[-1].append(article_score)
    return all_scores

def save_embeddings(articles_by_cat: list, tokenizer, model, device, path: str): 
    sentence_embeddings = list()
    document_embeddings = list()
    for cat_id, cat in enumerate(articles_by_cat): 
        sentence_embeddings.append([])
        document_embeddings.append([])
        for d_id in tqdm(range(0, len(cat))): 
            document = articles_by_cat[cat_id][d_id]
            sentences = nltk.sent_tokenize(document)
            sentence_embeddings[-1].append(get_embeddings(sentences, tokenizer, model, device))
            document_embeddings[-1].append(get_embeddings(document, tokenizer, model, device))
    
    sentence_embeddings_path = os.path.join(path, 'sentence_embeddings.pkl')
    pickle_dump(sentence_embeddings, sentence_embeddings_path)

    document_embeddings_path = os.path.join(path, 'document_embeddings.pkl')
    pickle_dump(document_embeddings, document_embeddings_path)

def pickle_load(path: str): 
    with open(path, 'rb') as f:    
        obj = pickle.load(f)
    return obj 

def pickle_dump(obj, path):
    with open(path, 'wb') as f: 
        pickle.dump(obj, f)

def json_dump(content, path):
    with open(path, 'w') as f: 
        json.dump(content, f, indent=4)
        
def generate_doc_ids_cat():
    doc_ids_cat = []

    doc_ids_cat.append([])
    for i in range(510):
        doc_ids_cat[-1].append(i)

    doc_ids_cat.append([])
    for i in range(510,893):
        doc_ids_cat[-1].append(i)

    doc_ids_cat.append([])
    for i in range(893, 1310):
        doc_ids_cat[-1].append(i)

    doc_ids_cat.append([])
    for i in range(1310, 1819):
        doc_ids_cat[-1].append(i)    

    doc_ids_cat.append([])
    for i in range(1819, 2220):
        doc_ids_cat[-1].append(i)
    
    return doc_ids_cat


def select_and_sort(scores: dict, o: str, p: int, l: int, sentence_lengths: list): 
    # Don't exceed maximum number of sentences. If it doesn't matter it should be set to 0
    max_elements = p if p else len(scores)
    sorted_scores = sort_by_value(scores, max_elements=max_elements, reverse=True)

    # Don't exceed maximum number of characters. If it doesn't matter it should be set to 0
    if l:
        total_length = 0
        cropped_sorted_scores = dict()
        for sent_number, score in sorted_scores.items():
            sent_length = sentence_lengths[sent_number]
            total_length += sent_length
            if total_length > l:
                break
            cropped_sorted_scores[sent_number] = score
        sorted_scores = cropped_sorted_scores

    if o == "rel": 
        return sorted_scores
    elif o == "app": 
        return sort_by_key(sorted_scores)
    else:
        return scores
