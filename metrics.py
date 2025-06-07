import numpy as np
from sklearn.metrics import roc_auc_score

def calculate_mrr(y_true, y_score):
    """
    평균 역순위(Mean Reciprocal Rank) 계산
    y_true: 실제 레이블 (첫 번째가 양성)
    y_score: 예측 점수
    """
    rankings = (-y_score).argsort()
    # 양성 샘플(첫 번째 아이템)의 순위 찾기
    rank = np.where(rankings == 0)[0][0] + 1
    return 1.0 / rank

def calculate_ndcg(y_true, y_score, k):
    """
    정규화된 누적 이득(Normalized Discounted Cumulative Gain) 계산
    """
    rankings = (-y_score).argsort()
    dcg_k = 0
    idcg_k = 0
    
    # DCG@K 계산
    for i in range(min(k, len(rankings))):
        if rankings[i] == 0:  # 양성 샘플이 상위 k개 안에 있는 경우
            dcg_k += 1.0 / np.log2(i + 2)
    
    # IDCG@K 계산 (이상적인 경우 양성 샘플이 첫 번째)
    idcg_k = 1.0  # 1 / log2(1 + 1)
    
    if idcg_k == 0:
        return 0
    
    return dcg_k / idcg_k

def calculate_metrics(y_true, y_score):
    """
    모든 평가 메트릭 계산
    """
    metrics = {}
    
    # AUC 계산
    metrics['auc'] = roc_auc_score(y_true, y_score)
    
    # 각 샘플에 대해 MRR과 NDCG 계산
    mrr_scores = []
    ndcg5_scores = []
    ndcg10_scores = []
    
    for i in range(len(y_true)):
        if isinstance(y_true[i], list):
            sample_true = y_true[i]
            sample_score = y_score[i]
        else:
            sample_true = y_true
            sample_score = y_score
        
        mrr_scores.append(calculate_mrr(sample_true, sample_score))
        ndcg5_scores.append(calculate_ndcg(sample_true, sample_score, 5))
        ndcg10_scores.append(calculate_ndcg(sample_true, sample_score, 10))
    
    metrics['mrr'] = np.mean(mrr_scores)
    metrics['ndcg5'] = np.mean(ndcg5_scores)
    metrics['ndcg10'] = np.mean(ndcg10_scores)
    
    return metrics 