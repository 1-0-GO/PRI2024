import numpy as np
from collections import defaultdict
import torch
import json
from G11_code.data_collection import map_path_to_articleID

def log10_tf(x):
    try:
        return (1 + np.log10(x)) 
    except ValueError:
        return 0
    
def tf_idf_term(N, df_t, tf_t_d):
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
        scores_for_term = np.log10(N/df_t) * np.outer(log10_tfs, log10_tfs)
        similarity_matrix += scores_for_term
        sq_norm_sentence += log10_tfs**2
    norm_sentence = np.sqrt(sq_norm_sentence)
    normalization = np.outer(norm_sentence, norm_sentence)
    return similarity_matrix / normalization

def sort_by_value(d: dict, max_elements: int, reverse=False) -> dict: 
    return dict(sorted(d.items(), key=lambda item: item[1], reverse=reverse)[:max_elements])

def sort_by_key(d: dict) -> dict: 
    return dict(sorted(d.items()))

def get_embedding(sentence: str, model, tokenizer, device, max_length=512) -> torch.tensor: 
    encoded_input = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=max_length)
    encoded_input.to(device)
    with torch.no_grad():
        output = model(**encoded_input)
    embedding = output["pooler_output"].squeeze()
    # mean pooled embedding might be better
    # mean_pooled_embedding = last_hidden_states.mean(axis=1)
    return embedding

def get_embeddings(sentences: list, model, tokenizer, device, max_length=512) -> list: 
    encoded_input = tokenizer(sentences, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
    encoded_input.to(device)
    output = model(**encoded_input)
    embedding = output["pooler_output"].squeeze()
    # mean pooled embedding might be better
    # mean_pooled_embedding = last_hidden_states.mean(axis=1)
    return embedding

def json_dump(content, path):
    with open(path, 'w') as f: 
        json.dump(content, f, indent=4)

def summary_compute(categorized_set_of_articles, summarization_function, path_to_articleID):
    all_scores = list()
    for category_id, category in enumerate(categorized_set_of_articles): 
        all_scores.append([])
        for cat_path in category: 
            article_id = path_to_articleID(cat_path)
            article_score = summarization_function(article_id)
            all_scores[-1].append(article_score)
    return all_scores