import torch
from torch.utils.data import DataLoader
from config import Config
from models.gerl import GERL
from data.mind_dataset import MINDDataset
from metrics import calculate_metrics
from tqdm import tqdm
import numpy as np
import random
import os
import time
from datetime import datetime, timedelta

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def prepare_batch(batch, device):
    """배치 데이터를 GPU로 이동"""
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

def train_epoch(model, train_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    total_samples = 0
    
    start_time = time.time()
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        batch = prepare_batch(batch, device)
        
        # Forward pass
        logits = model(batch)
        labels = batch['label']
        
        # Loss 계산
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 통계 업데이트
        total_loss += loss.item() * len(labels)
        total_samples += len(labels)
        
        # 진행 상황 표시
        avg_loss = total_loss / total_samples
        elapsed = time.time() - start_time
        speed = (batch_idx + 1) * len(batch) / elapsed
        eta = timedelta(seconds=int((len(train_loader) - batch_idx - 1) / speed))
        
        pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'speed': f'{speed:.1f} samples/s',
            'eta': str(eta)
        })
    
    return total_loss / total_samples

def evaluate(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_samples = 0
    
    pbar = tqdm(data_loader, desc='Evaluating')
    
    with torch.no_grad():
        for batch in pbar:
            batch = prepare_batch(batch, device)
            logits = model(batch)
            preds = torch.sigmoid(logits)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['label'].cpu().numpy())
            
            total_samples += len(batch['label'])
            pbar.set_postfix({'samples': total_samples})
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    metrics = calculate_metrics(all_labels, all_preds)
    return metrics

def main():
    # 설정
    config = Config()
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 데이터셋 로드
    print("\nLoading datasets...")
    train_dataset = MINDDataset(
        data_dir="MINDsmall_train",
        max_title_length=config.max_title_length,
        max_history_length=config.max_history_length,
        num_neighbors=config.max_neighbors
    )
    
    print(f"\nDataset statistics:")
    print(f"Total samples: {len(train_dataset)}")
    
    # 데이터 로더 생성
    print("\nCreating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    # 모델 초기화
    print("\nInitializing model...")
    model = GERL(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 옵티마이저 초기화
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # 학습 루프
    print("\nStarting training...")
    best_metric = 0
    start_time = time.time()
    
    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        
        # 학습
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch + 1)
        
        # 평가
        print("\nEvaluating...")
        metrics = evaluate(model, train_loader, device)
        
        # 결과 출력
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch+1}/{config.num_epochs} - Time: {timedelta(seconds=int(epoch_time))}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Evaluation Metrics:")
        print(f"AUC: {metrics['auc']:.4f}")
        print(f"MRR: {metrics['mrr']:.4f}")
        print(f"NDCG@5: {metrics['ndcg5']:.4f}")
        print(f"NDCG@10: {metrics['ndcg10']:.4f}")
        
        # 모델 저장
        if metrics['auc'] > best_metric:
            best_metric = metrics['auc']
            torch.save(model.state_dict(), 'best_model.pth')
            print("Saved new best model!")
        
        print("-" * 50)
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {timedelta(seconds=int(total_time))}")
    print(f"Best AUC: {best_metric:.4f}")

if __name__ == "__main__":
    main() 