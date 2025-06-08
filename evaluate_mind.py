"""
Section 4.2 "Performance Comparison"에서 설명된
MIND 데이터셋에서의 모델 성능 평가
"""

import torch
from torch.utils.data import DataLoader
from config import Config
from models.gerl import GERL
from data.mind_dataset import MINDDataset
from metrics import calculate_metrics
from tqdm import tqdm
import numpy as np

def prepare_batch(batch, device):
    """배치 데이터를 GPU로 이동"""
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

def evaluate(model, data_loader, device):
    """
    Section 4.2의 평가 지표로 모델 성능을 측정합니다.
    
    평가 지표:
    - AUC: 이진 분류 성능
    - MRR: 순위 기반 추천 정확도
    - nDCG@5/10: 상위 K개 추천의 순위 품질
    
    Args:
        model: 평가할 GERL 모델
        data_loader: 검증 데이터 로더
        device: 평가 장치 (CPU/GPU)
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="평가 진행 중"):
            batch = prepare_batch(batch, device)
            logits = model(batch)
            preds = torch.sigmoid(logits)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['label'].cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    metrics = calculate_metrics(all_labels, all_preds)
    return metrics

def main():
    """
    검증 데이터셋에서 모델 성능을 평가합니다.
    Section 4.1의 실험 설정을 사용합니다.
    """
    # 설정
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 Device: {device}")
    
    # 데이터셋 로드
    print("데이터셋 로딩...")
    dev_dataset = MINDDataset(
        data_dir="MINDsmall_dev",
        max_title_length=config.max_title_length,
        max_history_length=config.max_history_length,
        num_neighbors=config.max_neighbors
    )
    
    # 데이터 로더 생성
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # 모델 로드
    print("모델 로딩...")
    model = GERL(config).to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    
    # 평가
    print("평가 시작...")
    metrics = evaluate(model, dev_loader, device)
    
    print("\n평가 결과:")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"MRR: {metrics['mrr']:.4f}")
    print(f"NDCG@5: {metrics['ndcg5']:.4f}")
    print(f"NDCG@10: {metrics['ndcg10']:.4f}")

if __name__ == "__main__":
    main() 