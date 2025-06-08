import numpy as np
from sklearn.metrics import roc_auc_score

def compute_metrics(scores, labels):
    """Compute AUC, MRR, nDCG@5, nDCG@10
    
    Args:
        scores: Predicted scores [num_samples, num_candidates]
        labels: Ground truth labels [num_samples, num_candidates]
        
    Returns:
        dict: Dictionary containing metric values
    """
    metrics = {}
    
    # AUC
    auc_scores = []
    for score, label in zip(scores, labels):
        if len(np.unique(label)) > 1:  # Skip cases with all 0s or all 1s
            auc_scores.append(roc_auc_score(label, score))
    metrics['auc'] = np.mean(auc_scores)
    
    # MRR
    mrr_scores = []
    for score, label in zip(scores, labels):
        rank = np.where(np.argsort(-score) == np.where(label == 1)[0][0])[0][0] + 1
        mrr_scores.append(1.0 / rank)
    metrics['mrr'] = np.mean(mrr_scores)
    
    # nDCG@5
    ndcg_scores_5 = []
    for score, label in zip(scores, labels):
        idcg_5 = dcg_at_k(label, 5)
        if idcg_5 == 0:
            continue
        dcg_5 = dcg_at_k(label[np.argsort(-score)], 5)
        ndcg_scores_5.append(dcg_5 / idcg_5)
    metrics['ndcg@5'] = np.mean(ndcg_scores_5)
    
    # nDCG@10
    ndcg_scores_10 = []
    for score, label in zip(scores, labels):
        idcg_10 = dcg_at_k(label, 10)
        if idcg_10 == 0:
            continue
        dcg_10 = dcg_at_k(label[np.argsort(-score)], 10)
        ndcg_scores_10.append(dcg_10 / idcg_10)
    metrics['ndcg@10'] = np.mean(ndcg_scores_10)
    
    return metrics

def dcg_at_k(r, k):
    """Discounted Cumulative Gain at k
    
    Args:
        r: Relevance scores
        k: Number of items to consider
        
    Returns:
        float: DCG value
    """
    r = np.array(r)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.0 