import numpy as np
from collections import defaultdict
import torch
import json

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
        idf_term = np.log10(N/df_t)
        scores_for_term = idf_term * np.outer(log10_tfs, log10_tfs)
        similarity_matrix += scores_for_term
        sq_norm_sentence += idf_term * log10_tfs**2
    norm_sentence = np.sqrt(sq_norm_sentence)
    normalization = np.outer(norm_sentence, norm_sentence)
    return similarity_matrix / normalization

def sort_by_value(d: dict, max_elements: int, reverse=False) -> dict: 
    return dict(sorted(d.items(), key=lambda item: item[1], reverse=reverse)[:max_elements])

def sort_by_key(d: dict) -> dict: 
    return dict(sorted(d.items()))

def get_embeddings(sentences: list, tokenizer, model, device) -> np.array: 
    # Tokenization    
    tokenized = [tokenizer.encode(sent, add_special_tokens=True) for sent in sentences]  
    # Padding
    max_len = 0
    for i in tokenized:
        if len(i) > max_len:
            max_len = len(i)
    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized])
    # Masking
    attention_mask = np.where(padded != 0, 1, 0)
    # Running the model
    input_ids = torch.tensor(padded, device=device)
    attention_mask = torch.tensor(attention_mask, device=device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    # Get for all sentences (:), the CLS (0), from all hidden unit outputs (:) in the last hidden state
    features = outputs['last_hidden_state'][:,0,:].numpy()
    return features


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