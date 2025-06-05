#!/usr/bin/env python3
"""
빠른 테스트를 위한 스크립트
데이터 로딩과 모델 생성이 제대로 되는지 확인
"""

import torch
from models import GERL
from data.dataset import MINDDataset, create_data_loaders
from utils.config import Config

def test_data_loading():
    """데이터 로딩 테스트"""
    print("=" * 50)
    print("데이터 로딩 테스트")
    print("=" * 50)
    
    try:
        # 작은 데이터셋으로 테스트
        print("Train 데이터셋 로딩...")
        train_dataset = MINDDataset(split="train")
        print(f"✓ Train 데이터셋 크기: {len(train_dataset)}")
        
        print("\nDev 데이터셋 로딩...")
        dev_dataset = MINDDataset(split="dev")
        print(f"✓ Dev 데이터셋 크기: {len(dev_dataset)}")
        
        # 샘플 확인
        print("\n첫 번째 샘플 확인...")
        sample = train_dataset[0]
        print("샘플 키들:", list(sample.keys()))
        
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
        
        print("✓ 데이터 로딩 성공!")
        return True
        
    except Exception as e:
        print(f"✗ 데이터 로딩 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_creation():
    """모델 생성 테스트"""
    print("\n" + "=" * 50)
    print("모델 생성 테스트")
    print("=" * 50)
    
    try:
        # 설정
        config = Config()
        print("✓ 설정 생성 성공")
        
        # 모델 생성
        model = GERL(config)
        print("✓ 모델 생성 성공")
        
        # 파라미터 수 확인
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  총 파라미터 수: {total_params:,}")
        print(f"  훈련 가능한 파라미터 수: {trainable_params:,}")
        
        return True, model
        
    except Exception as e:
        print(f"✗ 모델 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_forward_pass():
    """Forward pass 테스트"""
    print("\n" + "=" * 50)
    print("Forward Pass 테스트")
    print("=" * 50)
    
    try:
        # 데이터 로더 생성
        train_loader, _ = create_data_loaders(batch_size=2)  # 작은 배치 크기
        
        # 모델 생성
        config = Config()
        model = GERL(config)
        
        # 첫 번째 배치 가져오기
        batch = next(iter(train_loader))
        print("✓ 배치 데이터 로드 성공")
        
        # Forward pass
        user_repr, news_repr = model(batch)
        print(f"✓ Forward pass 성공")
        print(f"  User representation shape: {user_repr.shape}")
        print(f"  News representation shape: {news_repr.shape}")
        
        # 손실 계산
        loss = model.compute_loss(batch)
        print(f"✓ 손실 계산 성공: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Forward pass 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 테스트 함수"""
    print("GERL 모델 빠른 테스트를 시작합니다...\n")
    
    # 1. 데이터 로딩 테스트
    data_success = test_data_loading()
    
    if not data_success:
        print("\n데이터 로딩에 실패했습니다. 데이터 파일을 확인해주세요.")
        return
    
    # 2. 모델 생성 테스트
    model_success, model = test_model_creation()
    
    if not model_success:
        print("\n모델 생성에 실패했습니다.")
        return
    
    # 3. Forward pass 테스트
    forward_success = test_forward_pass()
    
    if not forward_success:
        print("\nForward pass에 실패했습니다.")
        return
    
    print("\n" + "=" * 50)
    print("🎉 모든 테스트 통과!")
    print("이제 python train.py 명령으로 훈련을 시작할 수 있습니다.")
    print("=" * 50)

if __name__ == "__main__":
    main() 