import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import statistics
from G11_code.data_collection import flatten

def get_accuracy(num_sents, extracted_indices, relevant_indices):
    extracted_indices = set(extracted_indices)
    relevant_indices = set(relevant_indices)
    all_indices = set([i for i in range(num_sents)])
    extracted_negatives = all_indices.difference(extracted_indices)
    relevant_negatives = all_indices.difference(relevant_indices)
    return len(relevant_negatives.intersection(extracted_negatives)) + len(relevant_indices.intersection(extracted_indices))/num_sents
 
def get_precision(extracted_indices, relevant_indices):
    extracted_indices = set(extracted_indices)
    relevant_indices = set(relevant_indices)
    return len(relevant_indices.intersection(extracted_indices))/len(extracted_indices)

def get_recall(extracted_indices, relevant_indices):
    extracted_indices = set(extracted_indices)
    relevant_indices = set(relevant_indices)
    return len(relevant_indices.intersection(extracted_indices))/len(relevant_indices)

def get_F1(extracted_indices, relevant_indices):
    extracted_indices = set(extracted_indices)
    relevant_indices = set(relevant_indices)
    p = get_precision(extracted_indices, relevant_indices)
    r = get_recall(extracted_indices, relevant_indices)
    if p+r == 0: return 0
    return 2 * (p*r) / (p+r)

def plot_f1_per_category(f1_per_category, std_per_category, category_names, title):
# Choose a qualitative palette with 5 colors
    plt.style.use('ggplot')
    color = [mcolors.to_hex(c) for c in plt.cm.Set3(np.arange(5))]
    x = category_names
    plt.style.use('ggplot')
    color = [mcolors.to_hex(c) for c in plt.cm.Set3(np.arange(5))]
    plt.bar(x, f1_per_category, color=color)
    plt.errorbar(x, f1_per_category, std_per_category, fmt='.', color='Black', elinewidth=2,capthick=10,errorevery=1, alpha=0.5, ms=4, capsize = 2)
    plt.xlabel("Category")
    plt.ylabel("F1 Score")
    plt.ylim(0, 0.65)
    plt.title(title)
    plt.show()

def get_precision_recall_curve_data(extracted_indices, relevant_indices):
    extracted_indices = list(extracted_indices.keys())
    extracted_indices = list(map(int, extracted_indices))
    relevant_indices = set(relevant_indices)
    precision_level = list()
    recall_level = list()
    for k in range(len(extracted_indices)): 
        extracted_indices_at_k = set(extracted_indices[:k+1])
        intersect = extracted_indices_at_k.intersection(relevant_indices)
        precision = len(intersect)/len(extracted_indices_at_k)
        precision_level.append(precision)
        recall_level.append((k+1)/len(extracted_indices))
    return precision_level, recall_level

def plot_precision_recall_curve(precision_level, recall_level): 
    plt.plot(recall_level, precision_level)
    plt.xlim(0, 1)
    plt.ylim(0, 1.01)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall")
    plt.plot()


'''
evaluation(Sset,Rset,args)
    @input the set of summaries Sset produced from selected documents Dset ⊆ D
    (e.g. a single document, a category of documents, the whole collection),
    the corresponding reference extracts Rset, and optional arguments (evalu-
    ation, preprocessing, model options)

    @behavior assesses the produced summaries against the reference ones using the tar-
    get evaluation criteria

    @output evaluation statistics, including F-measuring at predefined p-or-l summary
    limits, recall-and-precision curves, MAP, and efficiency
'''
def evaluation(S: list, R: list, **args) -> list:
    precisions = list()
    recalls = list() 
    f1_scores = list()
    for category_id, (category_extracted_indices_rel, category_relevant_indices) in enumerate(zip(R, S)):
            precisions.append([])
            recalls.append([])
            f1_scores.append([])
            for extracted_indices_rel, relevant_indices in zip(category_extracted_indices_rel, category_relevant_indices):
                extracted_indices = list(extracted_indices_rel.keys())
                extracted_indices = list(map(int, extracted_indices))
                extracted_indices = extracted_indices[:args["p"]]
                precisions[-1].append(get_precision(extracted_indices, relevant_indices))
                recalls[-1].append(get_recall(extracted_indices, relevant_indices))
                f1_scores[-1].append(get_F1(extracted_indices, relevant_indices))
    
    mean_f1_per_catgory = list()
    mean_std_per_category = list()
    for f1 in f1_scores: 
        mean_f1_per_catgory.append(statistics.mean(f1))
        mean_std_per_category.append(statistics.stdev(f1))

    plot_f1_per_category(mean_f1_per_catgory, mean_std_per_category, args['category_names'], args['model_name'])

    mean_precision = statistics.mean(flatten(precisions))
    mean_recall = statistics.mean(flatten(recalls))
    mean_f1_scores = statistics.mean(flatten(f1_scores))
    
    # precision-recall curve only for one document 
    example_category = 0
    example_article = 1
    precision_c, recall_c = get_precision_recall_curve_data(R[example_category][example_article], S[example_category][example_article])
    plot_precision_recall_curve(precision_c, recall_c)
    mAP = statistics.mean(precision_c)
    print(f"Mean average precision: {mAP}")
    metrics = {'mean_precision': mean_precision, 'mean_recall': mean_recall, 'mean_f1_scores': mean_f1_scores}
    return metrics
