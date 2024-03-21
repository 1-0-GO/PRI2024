#!/usr/bin/env python
# coding: utf-8

# <H3>PRI 2023/24: first project delivery</H3>

# **GROUP 11**
# - Francisco Martins, 99068
# - Tunahan Güneş, 108108
# - Sebastian Weidinger, 111612

# <H3>Part I: demo of facilities</H3>

# In[224]:


import os
import json
from nltk.tokenize import sent_tokenize
import re


# In[225]:


def preprocess_article(text: str) -> str: 
    text = text.split("\n\n")
    # remove title
    text = text[1:]
    text = " ".join(text)
    text = text.strip()
    text = text.replace(" ?", "?")
    text = text.replace(" !", "!")
    text = re.sub(r'\s+', r' ', text)
    return text 


# In[226]:


def preprocess_summary(text: str) -> str: 
    text = text.replace(" ?", "?")
    text = text.replace(" !", "!")
    text = re.sub(r'\s+', r' ', text)
    return text


# In[227]:


def read_files(article_path, summary_path):
    articles = []
    summaries = []
    article_file_paths = []
    summary_file_paths = []
    category_names = list()
    for folder in os.listdir(article_path):
        category_names.append(folder)
        article_category_path = os.path.join(article_path, folder)
        summary_category_path = os.path.join(summary_path, folder)
        articles.append([])
        summaries.append([])
        article_file_paths.append([])
        summary_file_paths.append([])
        for file in os.listdir(article_category_path):
            article_file_path = os.path.join(article_category_path, file)
            summary_file_path = os.path.join(summary_category_path, file)
            article_file_paths[-1].append(article_file_path)
            summary_file_paths[-1].append(summary_file_path)
            # articles
            with open(article_file_path, "r", errors="ignore") as f:
                text = f.read()
                text = preprocess_article(text)
                articles[-1].append(text)
            #summaries 
            with open(summary_file_path, "r", errors="ignore") as f: 
                text = f.read()
                text = preprocess_summary(text)
                summaries[-1].append(text)
                
    print("Number of Categories:",len(os.listdir(article_path)))
    for i in range(len(os.listdir(article_path))):
        print("Number of Articles in", "'"+os.listdir(article_path)[i]+"'", "Category:",len(articles[i]))
    
    return article_file_paths, articles, summary_file_paths, summaries, category_names


# In[228]:


article_path = os.path.join("BBC News Summary", "BBC News Summary", "News Articles")
summary_path = os.path.join("BBC News Summary", "BBC News Summary", "Summaries")
print("Article path:", article_path)
print("Summary path:", summary_path)
article_file_paths, categorized_articles, summary_file_paths, categorized_summaries, category_names= read_files(article_path, summary_path)


# In[229]:


#Examplary text. The structure of the read file is: articles[category_no][document_no]. 
print(categorized_articles[0][0])
print(article_file_paths[508:512])
print(categorized_summaries[0][0])


# In[230]:


def flatten(lists) -> list: 
    return [element for sublist in lists for element in sublist]


# In[231]:


def get_summary_sentence_indices(articles: list, summaries: list) -> list: 
    categorized_summary_indices = list()
    categorized_article_summary = list(zip(articles, summaries))
    found_summary = 0
    faulty_summaries = list()
    for category_id, category in enumerate(categorized_article_summary): 
        article_summary_tuples = list(zip(category[0], category[1]))
        categorized_summary_indices.append([])
        for article_id, (article, summary) in enumerate(article_summary_tuples):
            sentence_indices = list()
            recreated_summary = ""
            article_sents = sent_tokenize(article)
            article_sents = set(article_sents)
            for sent_id, sent in enumerate(article_sents): 
                if summary.find(sent) != -1: 
                    sentence_indices.append(sent_id)
                    recreated_summary += sent
            categorized_summary_indices[-1].append(sentence_indices)
            summary_length = len(summary)
            recreated_summary_length = len(recreated_summary)
            if abs(summary_length - recreated_summary_length) < 3: 
                found_summary += 1
            else: 
                faulty_summaries.append((category_id, article_id))
    print(f"number of found summaries: {found_summary}")
    print(f"number of summaries: {len(flatten(summaries))}")
    print(f"{float(found_summary)/float(len(flatten(summaries))) * 100 :.2f}%")
    return categorized_summary_indices, faulty_summaries
                    


# In[232]:


#categorized_summary_sentence_indices, faulty_documents = get_summary_sentence_indices(categorized_articles, categorized_summaries)
categorized_summary_sentence_indices, faulty_summary_ids = get_summary_sentence_indices(categorized_articles, categorized_summaries)
print(categorized_summary_sentence_indices)
print(faulty_summary_ids)
#print(len(faulty_documents)/len(flatten(categorized_summary_sentence_indices)))


# In[233]:


def remove_entries(categorized_list: list, faulty_summary_ids: list): 
    for category_id, sent_id in faulty_summary_ids:
        del categorized_list[category_id][sent_id] 
    return categorized_list


# In[234]:


print(len(categorized_articles[1]))
categorized_articles = remove_entries(categorized_articles, faulty_summary_ids)
article_file_paths = remove_entries(article_file_paths, faulty_summary_ids)
print(len(categorized_articles[1]))
print(len(categorized_summaries[1]))
categorized_summaries = remove_entries(categorized_summaries, faulty_summary_ids)
summary_file_paths = remove_entries(summary_file_paths, faulty_summary_ids)
print(len(categorized_summaries[1]))


# In[182]:


#summary_sentence_indices = flatten(categorized_summary_sentence_indices)
with open("./testing/reference/categorized_summary_sentence_indices.json", "w") as f: 
    json.dump(categorized_summary_sentence_indices, f, indent=4)


# A) **Indexing** (preprocessing and indexing options)

# In[183]:


#code, statistics and/or charts here


# imports

# In[184]:


import time 
from typing import Union
import nltk
import numpy as np
import math
import torch
import sklearn
import json
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from collections import Counter, defaultdict
from tabulate import tabulate
from transformers import BertTokenizer, BertModel
from textblob import TextBlob

nltk.download('brown')


# In[185]:


# flatten list to get uncategorized collection
articles = flatten(categorized_articles)
summaries = flatten(categorized_summaries)
N = len(articles)
N_summaries = len(summaries)
#article_file_paths = flatten(article_file_paths)
dict_path_to_articleID = {path:i for i, path in enumerate(flatten(article_file_paths))}

def map_path_to_articleID(path):
    path = os.path.normpath(path)
    return dict_path_to_articleID.get(path)


# In[186]:


N


# In[187]:


N_summaries


# # Inverted Index Structure 
#  
# Each term points to a dictionary of document identifier and the term frequency in the document.
# 
# t1 -> [DF, {doc7: [TF_doc, ((s1, TF_sent), (s4, TF_sent), ...)], doc8: [TF_doc, ((s3, TF_sent), (s6, TF_sent), ...)], ...}]\
# t2 -> [DF, {doc300: [TF_doc, ((s1, TF_sent), (s5, TF_sent), ...)], doc401: [TF_doc, ((s5, TF_sent), (s4, TF_sent), ...)], ...}]\
# ...

# In[188]:


max_width = 20


# In[189]:


class TermFrequencies: 
    def __init__(self) -> None:
        self.tf_d_t = 0
        self.sent_tf = list()

    def add_sentence(self, sent_number, term_frequency):
        self.sent_tf.append((sent_number, term_frequency))
    
    def get_term_frequency(self):
        return self.tf_d_t
    
    def __repr__(self):
        padding = 5 - len(str(self.tf_d_t))
        return f'TF_d_t: {self.tf_d_t}{" " * padding}TF_per_sentence: {self.sent_tf}'


# In[190]:


class InvertedIndexEntry:
    def __init__(self) -> None:
        self.df_term = 0
        self.term_dict = defaultdict(TermFrequencies)
    
    def get_document(self, document):
        return self.term_dict.get(document, None)

    def get_or_default_document(self, document):
        return self.term_dict[document]
    
    def get_term_dict(self):
        return self.term_dict

    def update_document(self, document, new_value):
        self.term_dict[document] = new_value
    
    def __repr__(self):
        out = f'Document Frequency: {self.df_term}\n {" " * (max_width+2)} Term frequencies:\n'
        for doc_number, tfs in self.term_dict.items():
            padding = 5 - len(str(doc_number))
            out += f'{" " * (max_width + 3)} Doc {doc_number}{" " * padding}→ {tfs}\n'
        return out
    
    def calculate_df(self):
        self.df_term = len(self.term_dict)


# In[191]:


class InvertedIndex:
    def __init__(self, collection_size) -> None:
        self.inverted_index = defaultdict(InvertedIndexEntry)
        self.sentence_term_counts = list()
        self.sentence_num_chars = list()
        self.indexing_time = 0
        self.N = collection_size
    
    def __repr__(self):
        out = f'Time to index: {self.indexing_time}\nInverted Index:\n'
        for term, entry in self.inverted_index.items():
            padding = max_width - len(term)
            out += f'{term} {" " * padding} → {entry}\n'
        return out

    def get_or_default(self, term, document):
        return self.inverted_index[term].get_or_default_document(document)
    
    def update(self, term, document, new_value):
        self.inverted_index[term].update_document(document, new_value)
    
    def set_indexing_time(self, indexing_time):
        self.indexing_time = indexing_time
    
    def calculate_dfs(self):
        for entry in self.inverted_index.values():
            entry.calculate_df()  
    
    def get_sentence_lengths(self, document):
        return self.sentence_term_counts[document]

    def get_document_info(self, document):          
        info = {'Vocabulary': [], 'DF_t': [], 'TF_d_t': [], 'TF/sentence': []}
        for term, entry in self.inverted_index.items():
            doc_tfs = entry.get_document(document)
            if doc_tfs == None:
                continue
            info['Vocabulary'].append(term)
            info['DF_t'].append(entry.df_term)
            info['TF_d_t'].append(doc_tfs.tf_d_t)
            info['TF/sentence'].append(doc_tfs.sent_tf)
        return info
    
    def doc_to_string(self, document: int):
        out = f'Document id={document} → vocabulary and term frequencies:\n'
        info = self.get_document_info(document)
        table = zip(*info.values())
        headers = info.keys()
        return out + tabulate(table, headers, tablefmt="pretty")

    def get_term_info(self, term: str) -> dict: 
        entry = self.inverted_index[term]
        term_dict = entry.get_term_dict()
        term_info = dict()
        for(doc_id, tf) in term_dict.items(): 
            term_info[doc_id] = tf.get_term_frequency()
        return term_info



# In[192]:


def sentence_tokenize(sentence):
    sents = list()
    for paragraph in sentence.split('\n '):
        # split into sentences  
        sents_p = nltk.sent_tokenize(paragraph)
        for sent in sents_p:
            sents.append(sent)
    return sents


# In[193]:


'''
@ input a sentence to process, a tokenizer to split it into terms, a lemmatizer to normalize the terms,
a set consisting of stop_words to ignore

@behavior preprocesses the sentence

@output a triple consisting of the length in characters of the sentence, the number of terms in the sentence and
a list of terms and noun phrases appearing in the sentence (with repeated terms) 
'''
def preprocess(sentence: str, tokenizer: nltk.tokenize.api.TokenizerI, wnl: WordNetLemmatizer, stop_words=set):
    sent_out = list()
    tokenized_sentence = tokenizer.tokenize(sentence.lower())
    for term in tokenized_sentence:
        lem_term = wnl.lemmatize(term)
        if lem_term not in stop_words:      
            sent_out.append(lem_term)
    blobbed_sentence = TextBlob(sentence)
    all_noun_phrases = blobbed_sentence.noun_phrases
    # only include those that have multiple words
    noun_phrases = [n_p for n_p in all_noun_phrases if ' ' in n_p]
    sent_out.extend(noun_phrases)
    return len(sentence), len(tokenized_sentence), sent_out


# In[194]:


'''
indexing(D,args)
    @input document collection D and optional arguments on text preprocessing

    @behavior preprocesses the collection and, using existing libraries, 
    builds an inverted index with the relevant statistics for the subsequent summarization functions
    
    @output pair with the inverted index I and indexing time
'''
def indexing(articles, **args) -> InvertedIndex:
    start_time = time.time()
    inverted_index = InvertedIndex(len(articles))

    # tokenizer split words and keep hyphens e.g. state-of-the-art
    tokenizer = RegexpTokenizer(r'[\w|-]+')
    stop_words = set(stopwords.words('english'))
    wnl = WordNetLemmatizer()

    # loop through collection 
    for article_id, article in enumerate(articles): 
        sents = sentence_tokenize(article)
        # remove title (not needed for summarization task)
        #sents = sents[1:]
        preprocessing_results = [preprocess(sent, tokenizer, wnl, stop_words) for sent in sents]
        sent_num_chars, sent_term_counts, preprocessed_sentences = zip(*preprocessing_results)
        inverted_index.sentence_num_chars.append(list(sent_num_chars))
        inverted_index.sentence_term_counts.append(list(sent_term_counts))
        # count the term frequencies per sentence
        term_counter_per_sent = [Counter(sentence_terms) for sentence_terms in preprocessed_sentences]
        for sent_number, term_counter in enumerate(term_counter_per_sent):
            for term in term_counter: 
                tf = term_counter[term]
                term_document_tfs = inverted_index.get_or_default(term, article_id)
                term_document_tfs.tf_d_t += tf 
                term_document_tfs.add_sentence(sent_number, tf)
                inverted_index.update(term, article_id, term_document_tfs)
    inverted_index.calculate_dfs()
    end_time = time.time()
    indexing_time = end_time - start_time
    inverted_index.set_indexing_time(indexing_time)
    return inverted_index


# In[195]:


WordNetLemmatizer().lemmatize("played")


# In[196]:


terms = [str(i) for i in range(200000)]
sws = set(stopwords.words('english'))
[term in sws for term in terms]


# In[197]:


sent_removed = 'Hello, I\'m a data scientist and machine learning enthusiast. My name is John Williams and am a data scientist. The majestic, centuries-old oak tree stood proudly at the edge of the meadow'
blob = TextBlob(sent_removed)
[a for a in blob.noun_phrases if ' ' in a]


# In[198]:


s0 = 'Title. The little white little rabbit. The person played with the ball.'
s1 = 'Title. The white rabbit\'s ball. Rabbit rabbit ball rabbit.'
s2 = 'Title.  White, the little white rabbit. Little, little.'
test = [s0, s1, s2]
I_test = indexing(test)


# In[199]:


print(I_test)


# In[200]:


print(I_test.doc_to_string(2))


# In[250]:


I = indexing(articles)
t = 0


# In[35]:


print(I.sentence_term_counts[0:2])


# In[36]:


document_path = os.path.join("BBC News Summary", "BBC News Summary", "News Articles", "business", "509.txt")

print(I.doc_to_string(map_path_to_articleID(document_path)))


# # Summarization 
# 
# TF: 
# * Document: Term frequencies are assessed on document level.
# * Sentence: Term frequencies are assessed on sentence level.
# 
# IDF: Inverted document frequencies is assessed on collection level.\
# \
# Additional parameter "N" and "article_id". Is this allowed?
# 
# TODO: 
# * Evaluate choice and give reason: 
#     * IDF on document level?
#     * TF on document level for sentences? 
# * "order" parameter o
# * BM25
# * BERT embedding

# In[37]:


def log10_tf(x):
    try:
        return (1 + math.log10(x)) 
    except ValueError:
        return 0


# In[38]:


def tf_idf_term(N, df_t, tf_t_d):
    return log10_tf(tf_t_d) * math.log10(N/df_t)


# In[39]:


def BM25_term(df_t, tf_t_d, N, s_len_avg, s_len, k, b): 
    idf_t = math.log10(N/df_t)
    B = 1 - b + b * (s_len/s_len_avg)
    return idf_t * (tf_t_d * (k + 1))/(tf_t_d + k * B)


# In[40]:


def sort_by_value(d: dict, max_elements: int, reverse=False) -> dict: 
    return dict(sorted(d.items(), key=lambda item: item[1], reverse=reverse)[:max_elements])


# In[41]:


def max_by_value(d: dict) -> tuple:
    return max(d.items(), key=lambda item: item[1])


# In[42]:


def sort_by_key(d: dict) -> dict: 
    return dict(sorted(d.items()))


# In[43]:


def sort_by_order(scores: dict, o: str, p: int, l: int, sentence_lengths: list): 
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


# In[44]:


def get_embedding(sentence: str, model, tokenizer, device, max_length=512) -> torch.tensor: 
    encoded_input = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=max_length)
    encoded_input.to(device)
    output = model(**encoded_input)
    embedding = output["pooler_output"].squeeze()
    # mean pooled embedding might be better
    # mean_pooled_embedding = last_hidden_states.mean(axis=1)
    return embedding


# In[45]:


def get_embeddings(sentences: list, model, tokenizer, device, max_length=512) -> list: 
    encoded_input = tokenizer(sentences, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
    encoded_input.to(device)
    with torch.no_grad():
        output = model(**encoded_input)
    embedding = output["pooler_output"].squeeze()
    # mean pooled embedding might be better
    # mean_pooled_embedding = last_hidden_states.mean(axis=1)
    return embedding


# In[46]:


def tf_idf_relevance_scores(info: list, N):
    scores = defaultdict(int)
    sq_normalization_term = defaultdict(int)
    for _, df_t, tf_t_d, tf_per_sentence in info:
        rel_t_d = tf_idf_term(N, df_t, tf_t_d)
        for sent_number, tf_s_t in tf_per_sentence:
            scores[sent_number] += rel_t_d * log10_tf(tf_s_t)
            sq_normalization_term[sent_number] = log10_tf(tf_s_t)**2
    # normalization
    for sent_number, score in scores.items():
        scores[sent_number] = score / math.sqrt(sq_normalization_term[sent_number])
    return scores


# In[47]:


def tf_idf_sent_similarity():
    pass


# In[48]:


def update_info_after_sent_removal(info: list, sent_removed: int):
   for i, (term, df_t, tf_t_d, tf_per_sentence) in enumerate(info):
      info[i][2] -= tf_per_sentence[sent_removed]  
      info[i][3] = [(sent_number, score) for sent_number, score in info[i][3] if sent_number != sent_removed]
      if info[i][2] == 0:
         info[i][1] -= 1    


# In[49]:


'''
summarization(d,p,l,o,I,args)
    @input document d (the index in I/D), maximum number of sentences (p) and/or characters (l), order
    of presentation o (appearance in text vs relevance), inverted index I or the
    collection D, and optional arguments on IR models

    @behavior preprocesses d, assesses the relevance of each sentence in d against I ac-
    cording to args, and presents them in accordance with p, l and o
    
    @output summary s of document d, i.e. ordered pairs (sentence position in d, score)
'''
def summarization(d: int, p: int, l: int, o: int, I_or_D: Union[InvertedIndex, list], **args) -> list:

    ## if we receive the collection instead of the inverted index we must compute it first
    if type(I_or_D) == list:
        I = indexing(I_or_D)         
    else: 
        I = I_or_D
        
    doc_info = I.get_document_info(d)
    term_doc_info = zip(*doc_info.values())      
    sentence_lengths = I.get_sentence_lengths(d)
    scores = defaultdict(int)
    sq_normalization_term = defaultdict(int)

    if args['model'] == 'TF-IDF':
        scores = tf_idf_relevance_scores(term_doc_info, I.N)
    
    elif args['model'] == 'BM25':
        k = 0.2
        b = 0.75 
        avg_sentence_length = sum(sentence_lengths)/len(sentence_lengths)
        for term, df_t, tf_t_d, tf_per_sentence in term_doc_info: 
            for sent_number, tf_s_t in tf_per_sentence: 
                scores[sent_number] += BM25_term(df_t, tf_s_t, I.N, avg_sentence_length, sentence_lengths[sent_number], k, b)
    
    elif args['model'] == 'BERT':
        document = I_or_D[d]
        
        tokenizer = args['bert_tokenizer']
        bert_model = args['bert_model']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #bert_model.to(device)
        
        scores = defaultdict(float)
        # sentences 
        sentences = nltk.sent_tokenize(document)
        #sentences = sentences[1:]
        num_sentences = len(sentences)
        sent_embeddings = list()
        # every sentences on its own, no padding needed, faster on cpu
        # for gpu batches are better 
        bert_model.eval()
        for sent in sentences: 
            sent_embeddings.append(get_embeddings(sent, bert_model, tokenizer, device))
        #for sent in sentences: 
        #    sent_embedding = get_embedding(sent, bert_model, tokenizer, device)
        #    sent_embeddings.append(sent_embedding)
        # document
        doc_embedding = get_embedding(document, bert_model, tokenizer, device, max_length=256)
        for sent_id in range(0, num_sentences): 
            sent_vec = sent_embeddings[sent_id]
            score = torch.nn.functional.cosine_similarity(doc_embedding, sent_vec, dim=0)
            scores[sent_id] = score.item()
    
    elif args['model'] == 'MMR-TFIDF1':
        runs = p or len(scores)
        available_sentences = dict(scores)
        selected_sentences = defaultdict(int)
        for _ in range(runs):
            relevance_scores = tf_idf_relevance_scores(term_doc_info, I.N)
            sent_similarity = tf_idf_sent_similarity()
            best_sentence = max_by_value(scores)
        
    
    elif args['model'] == 'MMR-TFIDF2':
        runs = p or len(scores)
        available_sentences = dict(scores)
        selected_sentences = defaultdict(int)
        for _ in range(runs):
            relevance_scores = tf_idf_relevance_scores(term_doc_info, I.N)
            sent_similarity = tf_idf_sent_similarity()
            best_sentence = max_by_value(scores)
            
    else:
        raise ValueError("Currently we only support the following models:\n→ TF-IDF\n→ BM-25\n→ BERT\n→MMR-TFIDF")
    
    return sort_by_order(scores, o, p, l, sentence_lengths)


# In[50]:


article_id = map_path_to_articleID(document_path)
print("ORIGINAL DOCUMENT")
print(articles[article_id])
scores = summarization(d=article_id, p=7, l=1000, o="app", I_or_D=I, model='TF-IDF')

print("SUMMARY")
sentences = nltk.sent_tokenize(articles[article_id])
for sent_id, score in scores.items(): 
    print(f"{score:.2f}: {sentences[sent_id]}")


# In[51]:


article_id = map_path_to_articleID(document_path)
print("ORIGINAL DOCUMENT")
print(articles[article_id])
scores = summarization(d=article_id, p=5, l=1000, o="rel", I_or_D=I, model='BM25')

print("SUMMARY")
sentences = nltk.sent_tokenize(articles[article_id])
for sent_id, score in scores.items(): 
    print(f"{score:.2f}: {sentences[sent_id]}")


# In[52]:


bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
bert_model = BertModel.from_pretrained("bert-base-uncased")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model.to(device)

# In[53]:


article_id = map_path_to_articleID(document_path)
print("ORIGINAL DOCUMENT")
print(articles[article_id])
scores = summarization(d=article_id, p=5, l=1000, o="rel", I_or_D=articles, model='BERT', bert_model=bert_model, bert_tokenizer=bert_tokenizer)

print("SUMMARY")
sentences = nltk.sent_tokenize(articles[article_id])
for sent_id, score in scores.items(): 
    print(f"{score:.2f}: {sentences[sent_id]}")


# In[54]:


import json


# In[55]:




# In[56]:
from tqdm import tqdm

# In[57]:


# BERT
scores = list()
for category_id, category in enumerate(article_file_paths): 
    scores.append([])
    for path in tqdm(category): 
        article_id = map_path_to_articleID(path)
        article_score = summarization(d=article_id, p=7, l=1000, o="rel", I_or_D=articles, model='BERT', bert_model=bert_model, bert_tokenizer=bert_tokenizer)
        scores[-1].append(article_score)
with open('./testing/extracts/bert/categorized_scores.json', 'w') as f: 
    json.dump(scores, f, indent=4)


