#!/usr/bin/env python3
"""
빠른 테스트를 위한 스크립트
데이터 로딩과 모델 생성이 제대로 되는지 확인
"""

import torch
import sys
import os
from torch.utils.data import DataLoader

# 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import GERL
from data.dataset import MINDDataset
from utils.config import Config
from utils.metrics import evaluate_model, print_metrics

def test_data_loading():
    """데이터 로딩 테스트"""
    print("=" * 50)
    print("1. 데이터 로딩 테스트")
    print("=" * 50)
    
    try:
        # 작은 배치로 테스트
        dataset = MINDDataset("data/MIND_small", split="train", rebuild_graph=False)
        print(f"데이터셋 로드 성공: {len(dataset)} 샘플")
        print(f"어휘 크기: {len(dataset.vocab)}")
        print(f"그래프 - 뉴스 노드: {len(dataset.news_neighbors)}")
        print(f"그래프 - 사용자 노드: {len(dataset.user_neighbors)}")
        
        # 샘플 데이터 확인
        sample = dataset[0]
        print(f"\n샘플 데이터 구조:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape} {value.dtype}")
        
        return dataset
        
    except Exception as e:
        print(f"데이터 로딩 실패: {e}")
        return None

def test_model_creation(dataset):
    """모델 생성 테스트"""
    print("\n" + "=" * 50)
    print("2. 모델 생성 테스트")
    print("=" * 50)
    
    try:
        config = Config()
        
        # 실제 데이터 크기로 업데이트
        if dataset:
            config.update_vocab_sizes(
                vocab_size=len(dataset.vocab),
                num_users=10000,
                num_news=20000,
                num_topics=20
            )
        
        # 모델 생성
        model = GERL(config, vocab=dataset.vocab if dataset else None)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"모델 생성 성공")
        print(f"총 파라미터: {total_params:,}")
        print(f"학습 가능 파라미터: {trainable_params:,}")
        
        return model, config
        
    except Exception as e:
        print(f"모델 생성 실패: {e}")
        return None, None

def test_forward_pass(model, dataset, config):
    """순전파 테스트"""
    print("\n" + "=" * 50)
    print("3. 순전파 테스트")
    print("=" * 50)
    
    try:
        # 작은 배치 생성
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        batch = next(iter(dataloader))
        
        print(f"배치 크기: {batch['user_id'].shape[0]}")
        
        # 순전파
        model.eval()
        with torch.no_grad():
            user_repr, news_repr = model.forward(batch)
            predictions = model.predict(user_repr, news_repr)
            
        print(f"순전파 성공")
        print(f"사용자 표현: {user_repr.shape}")
        print(f"뉴스 표현: {news_repr.shape}")
        print(f"예측 점수: {predictions.shape}")
        print(f"예측 범위: [{predictions.min().item():.3f}, {predictions.max().item():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"순전파 실패: {e}")
        return False

def test_loss_computation(model, dataset):
    """손실 계산 테스트"""
    print("\n" + "=" * 50)
    print("4. 손실 계산 테스트")
    print("=" * 50)
    
    try:
        # 작은 배치 생성
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
        batch = next(iter(dataloader))
        
        # 손실 계산
        model.train()
        loss = model.compute_loss(batch)
        
        print(f"손실 계산 성공")
        print(f"손실 값: {loss.item():.4f}")
        print(f"손실이 유한한 값: {torch.isfinite(loss).item()}")
        
        return True
        
    except Exception as e:
        print(f"손실 계산 실패: {e}")
        return False

def test_gradient_computation(model, dataset):
    """기울기 계산 테스트"""
    print("\n" + "=" * 50)
    print("5. 기울기 계산 테스트")
    print("=" * 50)
    
    try:
        # 옵티마이저 생성
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 작은 배치 생성
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        batch = next(iter(dataloader))
        
        # 기울기 계산
        optimizer.zero_grad()
        loss = model.compute_loss(batch)
        loss.backward()
        
        # 기울기 확인
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
        
        print(f"기울기 계산 성공")
        print(f"기울기가 있는 파라미터 수: {len(grad_norms)}")
        print(f"평균 기울기 크기: {sum(grad_norms)/len(grad_norms):.6f}")
        print(f"최대 기울기 크기: {max(grad_norms):.6f}")
        
        return True
        
    except Exception as e:
        print(f"기울기 계산 실패: {e}")
        return False

def test_evaluation_metrics(model, dataset):
    """평가 지표 테스트"""
    print("\n" + "=" * 50)
    print("6. 평가 지표 테스트")
    print("=" * 50)
    
    try:
        # 작은 평가용 데이터로더
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
        
        # 몇 개 배치만 사용
        limited_data = []
        for i, batch in enumerate(dataloader):
            limited_data.append(batch)
            if i >= 3:  # 4개 배치만
                break
        
        # 임시 데이터로더 생성
        device = torch.device('cpu')  # CPU에서 테스트
        
        # 간단한 평가 수행
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in limited_data:
                user_repr, news_repr = model.forward(batch)
                predictions = model.predict(user_repr, news_repr)
                
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(batch['label'].cpu().numpy())
        
        import numpy as np
        all_predictions = np.concatenate(all_predictions)
        all_labels = np.concatenate(all_labels)
        
        print(f"평가 데이터 준비 성공")
        print(f"예측 수: {len(all_predictions)}")
        print(f"긍정 샘플: {np.sum(all_labels)}")
        print(f"부정 샘플: {len(all_labels) - np.sum(all_labels)}")
        
        return True
        
    except Exception as e:
        print(f"평가 지표 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 실행"""
    print("GERL 모델 통합 테스트 시작\n")
    
    # 1. 데이터 로딩
    dataset = test_data_loading()
    if not dataset:
        print("데이터 로딩 실패로 테스트 중단")
        return
    
    # 2. 모델 생성
    model, config = test_model_creation(dataset)
    if not model:
        print("모델 생성 실패로 테스트 중단")
        return
    
    # 3. 순전파
    if not test_forward_pass(model, dataset, config):
        print("순전파 실패로 테스트 중단")
        return
    
    # 4. 손실 계산
    if not test_loss_computation(model, dataset):
        print("손실 계산 실패로 테스트 중단")
        return
    
    # 5. 기울기 계산
    if not test_gradient_computation(model, dataset):
        print("기울기 계산 실패로 테스트 중단")
        return
    
    # 6. 평가 지표
    test_evaluation_metrics(model, dataset)
    
    print("\n" + "=" * 50)
    print("모든 테스트 완료!")
    print("=" * 50)
    print("데이터 로딩 및 그래프 구축")
    print("모델 아키텍처")
    print("순전파 및 역전파")
    print("손실 함수 계산")
    print("평가 지표 준비")
    print("\n이제 실제 훈련을 시작할 수 있습니다!")
    print("   python train.py --epochs 5 --batch_size 32")

if __name__ == "__main__":
    main() 