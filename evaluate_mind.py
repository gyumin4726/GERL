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
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
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
    # 설정
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 데이터셋 로드
    print("Loading dev dataset...")
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
    print("Loading model...")
    model = GERL(config).to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    
    # 평가
    print("Starting evaluation...")
    metrics = evaluate(model, dev_loader, device)
    
    print("\nEvaluation Results:")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"MRR: {metrics['mrr']:.4f}")
    print(f"NDCG@5: {metrics['ndcg5']:.4f}")
    print(f"NDCG@10: {metrics['ndcg10']:.4f}")

if __name__ == "__main__":
    main() 