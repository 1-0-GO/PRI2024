import numpy as np
import nltk
import time
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from collections import Counter, defaultdict
from tabulate import tabulate
from textblob import TextBlob

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
    
    def __repr__(self, max_width = 20):
        out = f'Document Frequency: {self.df_term}\n {" " * (max_width+2)} Term frequencies:\n'
        for doc_number, tfs in self.term_dict.items():
            padding = 5 - len(str(doc_number))
            out += f'{" " * (max_width + 3)} Doc {doc_number}{" " * padding}→ {tfs}\n'
        return out
    
    def calculate_df(self):
        self.df_term = len(self.term_dict)

class InvertedIndex:
    def __init__(self, collection_size, doc_lengths) -> None:
        self.inverted_index = defaultdict(InvertedIndexEntry)
        self.num_terms_in_sentences = list()
        self.sentence_num_chars = list()
        self.indexing_time = 0
        self.N = collection_size
        self.doc_lengths = np.array(doc_lengths)
    
    def __repr__(self, max_width = 20):
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
    
    def get_num_term_in_sentences(self, document):
        return self.num_terms_in_sentences[document]

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
    
def sentence_tokenize(sentence):
    sents = list()
    for paragraph in sentence.split('\n '):
        # split into sentences  
        sents_p = sent_tokenize(paragraph)
        for sent in sents_p:
            sents.append(sent)
    return sents

'''
@ input a sentence to process, a tokenizer to split it into terms, a lemmatizer to normalize the terms,
a set consisting of stop_words to ignore

@behavior preprocesses the sentence

@output a triple consisting of the length in characters of the sentence, the number of terms in the sentence and
a list of terms and noun phrases appearing in the sentence (with repeated terms) 
'''
def preprocess(sentence: str, tokenizer: nltk.tokenize.api.TokenizerI, lemmatizer, stop_words=set):
    sent_out = list()
    tokenized_sentence = tokenizer.tokenize(sentence.lower())
    for term in tokenized_sentence:
        lem_term = lemmatizer.lemmatize(term)
        if lem_term not in stop_words:      
            sent_out.append(lem_term)
    blobbed_sentence = TextBlob(sentence)
    all_noun_phrases = blobbed_sentence.noun_phrases
    # only include those that have multiple words
    noun_phrases = [n_p for n_p in all_noun_phrases if ' ' in n_p]
    sent_out.extend(noun_phrases)
    return len(sentence), len(tokenized_sentence), sent_out

'''
indexing(D,args)
    @input document collection D and optional arguments on text preprocessing

    @behavior preprocesses the collection and, using existing libraries, 
    builds an inverted index with the relevant statistics for the subsequent summarization functions
    
    @output pair with the inverted index I and indexing time
'''
def indexing(articles, **args) -> InvertedIndex:
    start_time = time.time()
    inverted_index = InvertedIndex(len(articles), [len(article) for article in articles])

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
        inverted_index.num_terms_in_sentences.append(list(sent_term_counts))
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