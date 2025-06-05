import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F


def dcg_at_k(relevance_scores, k):
    """Discounted Cumulative Gain at k"""
    relevance_scores = np.array(relevance_scores)[:k]
    if relevance_scores.size == 0:
        return 0.0
    
    # DCG = Σ(2^rel_i - 1) / log2(i + 1)
    gains = 2 ** relevance_scores - 1
    discounts = np.log2(np.arange(len(relevance_scores)) + 2)
    return (gains / discounts).sum()


def ndcg_at_k(relevance_scores, k):
    """Normalized Discounted Cumulative Gain at k"""
    dcg = dcg_at_k(relevance_scores, k)
    
    # IDCG: 이상적인 순서로 정렬했을 때의 DCG
    ideal_relevance = sorted(relevance_scores, reverse=True)
    idcg = dcg_at_k(ideal_relevance, k)
    
    if idcg == 0:
        return 0.0
    return dcg / idcg


def mean_reciprocal_rank(predictions, labels):
    """Mean Reciprocal Rank 계산"""
    mrr_scores = []
    
    for pred, label in zip(predictions, labels):
        # 예측 점수를 내림차순으로 정렬하여 순위 계산
        sorted_indices = np.argsort(pred)[::-1]
        
        # 첫 번째로 관련된 아이템의 위치 찾기
        for rank, idx in enumerate(sorted_indices, 1):
            if label[idx] == 1:  # 첫 번째 관련 아이템
                mrr_scores.append(1.0 / rank)
                break
        else:
            mrr_scores.append(0.0)  # 관련 아이템이 없는 경우
    
    return np.mean(mrr_scores)


def compute_auc(predictions, labels):
    """AUC 계산"""
    try:
        # Flatten arrays if needed
        if len(predictions.shape) > 1:
            predictions = predictions.flatten()
        if len(labels.shape) > 1:
            labels = labels.flatten()
        
        return roc_auc_score(labels, predictions)
    except ValueError:
        # AUC를 계산할 수 없는 경우 (모든 레이블이 같은 값인 경우)
        return 0.0


def compute_ndcg(predictions, labels, k_values=[5, 10]):
    """nDCG@k 계산"""
    ndcg_scores = {}
    
    for k in k_values:
        ndcg_list = []
        
        for pred, label in zip(predictions, labels):
            # 예측 점수를 내림차순으로 정렬
            sorted_indices = np.argsort(pred)[::-1]
            
            # 정렬된 순서에 따른 실제 레이블
            sorted_labels = [label[idx] for idx in sorted_indices]
            
            # nDCG 계산
            ndcg = ndcg_at_k(sorted_labels, k)
            ndcg_list.append(ndcg)
        
        ndcg_scores[f'nDCG@{k}'] = np.mean(ndcg_list)
    
    return ndcg_scores


def evaluate_model(model, data_loader, device):
    """모델 평가"""
    model.eval()
    
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in data_loader:
            # 데이터를 device로 이동
            for key in batch:
                batch[key] = batch[key].to(device)
            
            # 예측
            user_repr, news_repr = model.forward(batch)
            predictions = model.predict(user_repr, news_repr)
            
            # 손실 계산
            loss = model.compute_loss(batch)
            total_loss += loss.item()
            num_batches += 1
            
            # CPU로 이동하여 저장
            predictions_cpu = predictions.cpu().numpy()
            labels_cpu = batch['label'].cpu().numpy()
            
            all_predictions.append(predictions_cpu)
            all_labels.append(labels_cpu)
    
    # 모든 예측과 레이블을 하나로 합침
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    
    # 각 지표 계산
    metrics = {}
    
    # AUC 계산
    auc = compute_auc(all_predictions, all_labels)
    metrics['AUC'] = auc
    
    # 평균 손실
    metrics['Loss'] = total_loss / num_batches
    
    # 개별 impression에 대한 MRR과 nDCG는 배치 레벨에서 계산
    # 실제 구현에서는 impression 단위로 그룹화해야 함
    # 여기서는 간단한 버전으로 구현
    
    return metrics


def evaluate_model_detailed(model, data_loader, device, impression_groups=None):
    """상세한 모델 평가 (impression 단위)"""
    model.eval()
    
    impression_predictions = []  # impression별 예측 점수들
    impression_labels = []       # impression별 실제 레이블들
    total_loss = 0.0
    num_batches = 0
    
    current_impression = []
    current_labels = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # 데이터를 device로 이동
            for key in batch:
                batch[key] = batch[key].to(device)
            
            # 예측
            user_repr, news_repr = model.forward(batch)
            predictions = model.predict(user_repr, news_repr)
            
            # 손실 계산
            loss = model.compute_loss(batch)
            total_loss += loss.item()
            num_batches += 1
            
            # CPU로 이동
            predictions_cpu = predictions.cpu().numpy()
            labels_cpu = batch['label'].cpu().numpy()
            
            # 배치의 각 샘플을 처리
            for pred, label in zip(predictions_cpu, labels_cpu):
                current_impression.append(pred)
                current_labels.append(label)
                
                # impression이 완료되었는지 확인 (실제로는 impression_id로 그룹화)
                # 여기서는 간단히 배치 크기만큼 모였을 때 하나의 impression으로 간주
                if len(current_impression) >= 10:  # 임의의 impression 크기
                    impression_predictions.append(current_impression.copy())
                    impression_labels.append(current_labels.copy())
                    current_impression.clear()
                    current_labels.clear()
    
    # 남은 데이터 처리
    if current_impression:
        impression_predictions.append(current_impression)
        impression_labels.append(current_labels)
    
    # 각 지표 계산
    metrics = {}
    
    # AUC 계산 (전체 데이터에 대해)
    all_preds = np.concatenate([np.array(imp) for imp in impression_predictions])
    all_labels = np.concatenate([np.array(imp) for imp in impression_labels])
    
    if len(np.unique(all_labels)) > 1:
        metrics['AUC'] = compute_auc(all_preds, all_labels)
    else:
        metrics['AUC'] = 0.0
    
    # MRR 계산 (impression 단위)
    if impression_predictions:
        mrr = mean_reciprocal_rank(impression_predictions, impression_labels)
        metrics['MRR'] = mrr
    else:
        metrics['MRR'] = 0.0
    
    # nDCG 계산 (impression 단위)
    if impression_predictions:
        ndcg_scores = compute_ndcg(impression_predictions, impression_labels, k_values=[5, 10])
        metrics.update(ndcg_scores)
    else:
        metrics['nDCG@5'] = 0.0
        metrics['nDCG@10'] = 0.0
    
    # 평균 손실
    metrics['Loss'] = total_loss / num_batches
    
    return metrics


def print_metrics(metrics, prefix=""):
    """평가 지표 출력"""
    if prefix:
        print(f"\n{prefix} Metrics:")
    else:
        print("\nEvaluation Metrics:")
    
    print("-" * 40)
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric_name:12}: {value:.4f}")
        else:
            print(f"{metric_name:12}: {value}")
    print("-" * 40)


def compare_metrics(metrics1, metrics2, names=["Model 1", "Model 2"]):
    """두 모델의 지표 비교"""
    print(f"\nModel Comparison: {names[0]} vs {names[1]}")
    print("=" * 60)
    print(f"{'Metric':<12} {'Model 1':<12} {'Model 2':<12} {'Improvement':<12}")
    print("-" * 60)
    
    for metric in metrics1.keys():
        if metric in metrics2:
            val1 = metrics1[metric]
            val2 = metrics2[metric]
            if isinstance(val1, float) and isinstance(val2, float):
                improvement = ((val2 - val1) / val1 * 100) if val1 != 0 else 0
                print(f"{metric:<12} {val1:<12.4f} {val2:<12.4f} {improvement:+7.2f}%")
            else:
                print(f"{metric:<12} {val1:<12} {val2:<12} {'N/A':<12}")
    print("-" * 60) 