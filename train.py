import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
from tqdm import tqdm

from models import GERL
from data.dataset import create_data_loaders
from utils.config import Config
from utils.metrics import evaluate_model, print_metrics


def train_epoch(model, train_loader, optimizer, device, config):
    """한 에포크 훈련"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch in progress_bar:
        # 데이터를 device로 이동
        for key in batch:
            batch[key] = batch[key].to(device)
        
        optimizer.zero_grad()
        
        # 순전파 및 손실 계산
        loss = model.compute_loss(batch)
        
        # 역전파
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # 진행 상황 업데이트
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate_epoch(model, dev_loader, device):
    """평가"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="Evaluating"):
            # 데이터를 device로 이동
            for key in batch:
                batch[key] = batch[key].to(device)
            
            # 손실 계산
            loss = model.compute_loss(batch)
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/MIND_small', help='Data directory')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--save_dir', default='saved_models', help='Model save directory')
    
    args = parser.parse_args()
    
    # 설정
    config = Config(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr
    )
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 설정 출력
    config.display()
    
    # 데이터 로더 생성
    print("Creating data loaders...")
    train_loader, dev_loader = create_data_loaders(
        data_dir=args.data_dir, 
        batch_size=config.batch_size
    )
    
    # 첫 번째 배치에서 실제 데이터 크기 확인
    sample_batch = next(iter(train_loader))
    print("\n데이터 배치 정보:")
    for key, value in sample_batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # 모델 생성
    print("\nCreating model...")
    model = GERL(config).to(device)
    
    # 옵티마이저
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # 모델 저장 디렉토리 생성
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 훈련 루프
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        print(f"\n========== Epoch {epoch+1}/{config.num_epochs} ==========")
        
        # 훈련
        train_loss = train_epoch(model, train_loader, optimizer, device, config)
        print(f"Train Loss: {train_loss:.4f}")
        
        # 평가
        val_loss = evaluate_epoch(model, dev_loader, device)
        print(f"Val Loss: {val_loss:.4f}")
        
        # 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(args.save_dir, 'gerl_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, model_path)
            print(f"Best model saved: {model_path}")
        
        # 매 에포크마다 체크포인트 저장
        checkpoint_path = os.path.join(args.save_dir, f'gerl_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': config
        }, checkpoint_path)
    
    print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main() 