import spacy
def feature_extraction(s: str, d: str, **args):
        
    #Getting the positional features of the sentence
    d_paragraphs = d.split("\n\n")[1:]
    
    s_features = dict()
    sent_no = 0
    found= False
    for paragraph_no in range(len(d_paragraphs)): 
        sents_p = nltk.sent_tokenize(d_paragraphs[paragraph_no])
        for in_parag_no in range(len(sents_p)):
            if s == sents_p[in_parag_no]:
                s_features['paragraph_no'] = paragraph_no
                s_features['position_in_paragraph'] = in_parag_no
                s_sent_id = sent_no
                found = True
                break
            sent_no +=1
        if found:
            break

    #Calculating the scores for the document
    d_scores_bert = summarization(d=0, p=0, l=0, o='rel', I_or_D=[d], model='BERT')
    d_scores_tfidf = summarization(d=0, p=0, l=0, o='rel', I_or_D=[d], model='TF-IDF')
    d_scores_bm25 = summarization(d=0, p=0, l=0, o='rel',I_or_D=[d], model='BM25')
    
    #Calculating the ranks for the document
    d_ranks_bert = {key: rank for rank, key in enumerate(d_scores_bert)}
    d_ranks_tfidf = {key: rank for rank, key in enumerate(d_scores_tfidf)}
    d_ranks_bm25 = {key: rank for rank, key in enumerate(d_scores_bm25)}
            
    #Setting positional & ranking features
    s_features = dict()
    s_features['position'] = s_sent_id
    s_features['score_bert'] = d_scores_bert[s_sent_id]
    s_features['score_tfidf'] = d_scores_tfidf[s_sent_id]
    s_features['score_bm25'] = d_scores_bm25[s_sent_id]
    s_features['rank_bert'] = d_ranks_bert[s_sent_id]
    s_features['rank_tfidf'] = d_ranks_tfidf[s_sent_id]
    s_features['rank_bm25'] = d_ranks_bm25[s_sent_id]
    
    #Calculate named entity features
    nlp = spacy.load('en_core_web_sm')
    possible_ent_labels = nlp.get_pipe("ner").labels
    s_ent_labels = [ent.label_ for ent in nlp(s).ents]
    s_ent_counts = {string: 0 for string in possible_ent_labels}
    for string in s_ent_labels:
        if string in s_ent_counts:
            s_ent_counts[string] += 1
    s_ent_counts = { (lambda k: k.lower() + '_count')(k) : v for k, v in s_ent_counts.items() }
            
    
    #Merging the the two features
    s_features.update(s_ent_counts)
    
    return s_features

