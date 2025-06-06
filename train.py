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
    """평가 (논문에서 사용한 지표들로)"""
    # 논문의 평가 지표 사용
    metrics = evaluate_model(model, dev_loader, device)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/MIND_small', help='Data directory')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--save_dir', default='saved_models', help='Model save directory')
    
    args = parser.parse_args()
    
    print(" GERL Training")
    print("=" * 50)
    
    # 그래프 파일 존재 여부 확인 (small 버전 우선)
    vocab_small_path = os.path.join(args.data_dir, "vocab_small.pkl")
    train_graph_small_path = os.path.join(args.data_dir, "graph_train_small.pkl")
    vocab_path = os.path.join(args.data_dir, "vocab.pkl")
    train_graph_path = os.path.join(args.data_dir, "graph_train.pkl")
    
    # Small 버전이 있으면 우선 사용
    if os.path.exists(vocab_small_path) and os.path.exists(train_graph_small_path):
        print("Using small pre-built graph files for fast loading")
        print(f"   Vocab: {vocab_small_path}")
        print(f"   Train graph: {train_graph_small_path}")
        use_small = True
    elif os.path.exists(vocab_path) and os.path.exists(train_graph_path):
        print("Using standard pre-built graph files")
        print(f"   Vocab: {vocab_path}")
        print(f"   Train graph: {train_graph_path}")
        use_small = False
    else:
        print(" Pre-built graph files not found!")
        print("For optimal training performance, please run:")
        print(f"   python build_graph.py --small  (recommended, faster)")
        print(f"   python build_graph.py          (full dataset)")
        print()
        print("Continuing with in-memory graph building (slower)...")
        use_small = False
    print()
    
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
        batch_size=config.batch_size,
        rebuild_graph=False
    )
    
    # 첫 번째 배치에서 실제 데이터 크기 확인
    sample_batch = next(iter(train_loader))
    print("\n데이터 배치 정보:")
    for key, value in sample_batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # 어휘 정보 가져오기
    train_dataset = train_loader.dataset
    vocab = train_dataset.vocab
    
    # 설정 업데이트 (실제 데이터 크기 반영)
    config.update_vocab_sizes(
        vocab_size=len(vocab),
        num_users=50000,  # 실제 사용자 수로 업데이트 가능
        num_news=100000,  # 실제 뉴스 수로 업데이트 가능
        num_topics=20
    )
    
    # 모델 생성 (어휘 정보 전달)
    print("\nCreating model...")
    model = GERL(config, vocab=vocab).to(device)
    
    # 옵티마이저
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # 모델 저장 디렉토리 생성
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 훈련 루프
    best_val_auc = 0.0  # AUC가 높을수록 좋음
    
    for epoch in range(config.num_epochs):
        print(f"\n========== Epoch {epoch+1}/{config.num_epochs} ==========")
        
        # 훈련
        train_loss = train_epoch(model, train_loader, optimizer, device, config)
        print(f"Train Loss: {train_loss:.4f}")
        
        # 평가 (논문의 지표 사용)
        val_metrics = evaluate_epoch(model, dev_loader, device)
        print_metrics(val_metrics, "Validation")
        
        val_auc = val_metrics.get('AUC', 0.0)
        val_loss = val_metrics.get('Loss', float('inf'))
        
        # 모델 저장 (AUC 기준)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            model_path = os.path.join(args.save_dir, 'gerl_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': config,
                'vocab': vocab
            }, model_path)
            print(f"Best model saved (AUC: {val_auc:.4f}): {model_path}")
        
        # 매 에포크마다 체크포인트 저장
        checkpoint_path = os.path.join(args.save_dir, f'gerl_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_metrics': val_metrics,
            'config': config,
            'vocab': vocab
        }, checkpoint_path)
    
    print(f"\nTraining completed! Best validation AUC: {best_val_auc:.4f}")


if __name__ == "__main__":
    main() 