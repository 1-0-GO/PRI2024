import spacy

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
            
def count_keyword_occurances(sentence:str, keywords:list):
    # Count occurrences of words with a preceding space or at the beginning of the sentence
    return sum(sentence.lower().count((' ' + word).lower()) for word in keywords)            

def feature_extraction(s: int, d: int, **args):
    
    #Getting the required args values
    article_file_paths = ('article_file_paths' in args and args['article_file_paths']) or None
    articles = ('articles' in args and args['articles']) or None
    #The 'scores' in args has to be a tuple that contains all scores from BERT, TF-IDF, BM-25 in that order.
    scores = ('scores' in args and args['scores']) or None
    keywords = ('keywords' in args and args['keywords']) or None
    
    #Calculating the ranks
    ranks_bert = {key: rank for rank, key in enumerate(scores[0][d])} #The first one has to bert scores
    ranks_tfidf = {key: rank for rank, key in enumerate(scores[1][d])} #The second one has to tf-idf scores
    ranks_bm25 = {key: rank for rank, key in enumerate(scores[2][d])} #The third one has to bm-25 scores
    
    #Getting the textual sentence to count keywords in the 
    sentence = nltk.sent_tokenize(articles[d])[s]
    
    s_features = dict()
    s_features['document_id'] = d
    s_features['sent_id'] = s
    s_features['paragraph_no'], s_features['position_in_paragraph'] = get_paragraph_features(article_file_paths[d], s)
    s_features['score_bert'] = scores[0][d][s]
    s_features['score_tfidf'] = scores[1][d][s]
    s_features['score_bm25'] = scores[2][d][s]
    s_features['rank_bert'] = ranks_bert[s]
    s_features['rank_tfidf'] = ranks_tfidf[s]
    s_features['rank_bm25'] = ranks_bm25[s]
    s_features['keyword_count'] = count_keyword_occurances(sentence, keywords=keywords[d].keys())
    nlp = spacy.load('en_core_web_sm')
    possible_ent_labels = nlp.get_pipe("ner").labels
    for string in s_ent_labels:
        if string in s_ent_counts:
            s_ent_counts[string] += 1
    s_ent_counts = { (lambda k: k.lower() + '_count')(k) : v for k, v in s_ent_counts.items() }
    s_features.update(s_ent_counts)
  
    return s_features