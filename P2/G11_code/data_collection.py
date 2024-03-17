import os
from nltk.tokenize import sent_tokenize
import re

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

def preprocess_summary(text: str) -> str: 
    text = text.replace(" ?", "?")
    text = text.replace(" !", "!")
    text = re.sub(r'\s+', r' ', text)
    return text

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

def flatten(lists) -> list: 
    return [element for sublist in lists for element in sublist]

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
                    
def remove_entries(categorized_list: list, faulty_summary_ids: list):
    def remove_entries_by_category(category_contents, category_id):
        return [content for article_id, content in enumerate(category_contents) if (category_id, article_id) not in faulty_summary_ids]
    return [remove_entries_by_category(contents, category_id) for category_id, contents in enumerate(categorized_list)]

def map_path_to_articleID(path, article_file_paths):
    dict_path_to_articleID = {path:i for i, path in enumerate(article_file_paths)}
    path = os.path.normpath(path)
    return dict_path_to_articleID.get(path)
