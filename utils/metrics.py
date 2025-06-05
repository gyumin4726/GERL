import numpy as np
from sklearn.metrics import roc_auc_score
import torch


def dcg_score(y_true, y_score, k=10):
    """Discounted Cumulative Gain"""
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    """Normalized Discounted Cumulative Gain"""
    actual = dcg_score(y_true, y_score, k)
    best = dcg_score(y_true, y_true, k)
    return actual / best if best > 0 else 0.0


def mrr_score(y_true, y_score):
    """Mean Reciprocal Rank"""
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    
    # 첫 번째 relevant item의 위치 찾기
    for i, relevant in enumerate(y_true):
        if relevant == 1:
            return 1.0 / (i + 1)
    return 0.0


def auc_score(y_true, y_score):
    """Area Under Curve"""
    if len(np.unique(y_true)) == 1:
        return 0.5  # 모든 라벨이 같은 경우
    return roc_auc_score(y_true, y_score)


def evaluate_metrics(y_true_list, y_score_list):
    """
    여러 impression에 대한 평가 지표 계산
    
    Args:
        y_true_list: List of true labels for each impression
        y_score_list: List of predicted scores for each impression
    
    Returns:
        dict: 평가 지표들의 딕셔너리
    """
    
    auc_scores = []
    mrr_scores = []
    ndcg5_scores = []
    ndcg10_scores = []
    
    for y_true, y_score in zip(y_true_list, y_score_list):
        # AUC 계산
        auc_scores.append(auc_score(y_true, y_score))
        
        # MRR 계산
        mrr_scores.append(mrr_score(y_true, y_score))
        
        # nDCG@5 계산
        ndcg5_scores.append(ndcg_score(y_true, y_score, k=5))
        
        # nDCG@10 계산
        ndcg10_scores.append(ndcg_score(y_true, y_score, k=10))
    
    # 평균 계산
    metrics = {
        'AUC': np.mean(auc_scores),
        'MRR': np.mean(mrr_scores),
        'nDCG@5': np.mean(ndcg5_scores),
        'nDCG@10': np.mean(ndcg10_scores)
    }
    
    return metrics


def evaluate_model(model, data_loader, device):
    """
    모델 평가 함수
    
    Args:
        model: 평가할 모델
        data_loader: 평가 데이터 로더
        device: 계산 장치
    
    Returns:
        dict: 평가 지표들
    """
    model.eval()
    
    all_y_true = []
    all_y_score = []
    
    with torch.no_grad():
        for batch in data_loader:
            # 배치 데이터를 device로 이동
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # 모델 예측
            user_repr, news_repr = model(batch)
            scores = model.predict(user_repr, news_repr)
            
            # impression별로 데이터 분리
            impression_ids = batch.get('impression_id', torch.arange(len(scores)))
            labels = batch.get('label', torch.ones_like(scores))  # 기본값으로 1 설정
            
            # impression별로 그룹화
            unique_impressions = torch.unique(impression_ids)
            
            for imp_id in unique_impressions:
                mask = impression_ids == imp_id
                imp_scores = scores[mask].cpu().numpy()
                imp_labels = labels[mask].cpu().numpy()
                
                all_y_true.append(imp_labels)
                all_y_score.append(imp_scores)
    
    # 평가 지표 계산
    metrics = evaluate_metrics(all_y_true, all_y_score)
    
    return metrics


def print_metrics(metrics, title="Evaluation Results"):
    """평가 지표 출력"""
    print(f"\n{title}")
    print("=" * 50)
    print(f"AUC:      {metrics['AUC']:.4f}")
    print(f"MRR:      {metrics['MRR']:.4f}")
    print(f"nDCG@5:   {metrics['nDCG@5']:.4f}")
    print(f"nDCG@10:  {metrics['nDCG@10']:.4f}")
    print("=" * 50) 