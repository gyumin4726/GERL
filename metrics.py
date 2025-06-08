"""
Section 4.1 "Datasets and Experimental Settings"에서 사용된
평가 메트릭 구현

이 모듈은 뉴스 추천 시스템의 성능을 평가하기 위한
다양한 메트릭을 구현합니다.

주요 메트릭:
1. AUC: 클릭/비클릭 분류의 성능
2. MRR: 순위 예측의 정확도
3. nDCG@5, nDCG@10: 상위 K개 추천의 품질
"""

import numpy as np
from sklearn.metrics import roc_auc_score

def compute_metrics(scores, labels):
    """
    Section 4.1에서 설명된 평가 메트릭 계산
    
    모든 인상(impression)에 대해 평균 AUC, MRR, nDCG@5, nDCG@10
    점수를 계산합니다.
    
    Args:
        scores: 예측 점수 [num_samples, num_candidates]
        labels: 실제 레이블 [num_samples, num_candidates]
    """
    metrics = {}
    
    # AUC: 클릭/비클릭 분류의 성능 측정
    auc_scores = []
    for score, label in zip(scores, labels):
        if len(np.unique(label)) > 1:  # 모두 0이나 1인 경우 제외
            auc_scores.append(roc_auc_score(label, score))
    metrics['auc'] = np.mean(auc_scores)
    
    # MRR: 첫 번째 관련 항목의 순위 역수의 평균
    mrr_scores = []
    for score, label in zip(scores, labels):
        rank = np.where(np.argsort(-score) == np.where(label == 1)[0][0])[0][0] + 1
        mrr_scores.append(1.0 / rank)
    metrics['mrr'] = np.mean(mrr_scores)
    
    # nDCG@5: 상위 5개 추천의 품질
    ndcg_scores_5 = []
    for score, label in zip(scores, labels):
        idcg_5 = dcg_at_k(label, 5)
        if idcg_5 == 0:
            continue
        dcg_5 = dcg_at_k(label[np.argsort(-score)], 5)
        ndcg_scores_5.append(dcg_5 / idcg_5)
    metrics['ndcg@5'] = np.mean(ndcg_scores_5)
    
    # nDCG@10: 상위 10개 추천의 품질
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
    """
    Discounted Cumulative Gain 계산
    
    순위에 따라 가중치를 부여하여 추천의 품질을 평가합니다.
    상위 순위의 관련 항목에 더 높은 가중치를 부여합니다.
    
    Args:
        r: 관련성 점수 리스트
        k: 고려할 상위 항목 수
    """
    r = np.array(r)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.0 