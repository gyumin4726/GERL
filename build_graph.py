#!/usr/bin/env python3
"""
GERL 모델을 위한 이분 그래프 구축 스크립트

이 스크립트는 MIND 데이터셋으로부터 사용자-뉴스 이분 그래프를 구축합니다.
논문에 따라:
- "동일한 사용자가 본 뉴스들이 이웃 뉴스"
- "동일한 뉴스를 클릭한 사용자들이 이웃 사용자"

사용법:
    python build_graph.py --data_dir data/MIND_small
"""

import pandas as pd
import pickle
import os
import argparse
from collections import defaultdict
from tqdm import tqdm
import time


def load_mind_data(data_dir, split="train"):
    """MIND 데이터 로드"""
    print(f"Loading {split} data from {data_dir}...")
    
    # 뉴스 데이터 로드
    news_path = f"{data_dir}/{split}/news.tsv"
    print(f"   Loading news data: {news_path}")
    news_df = pd.read_csv(
        news_path,
        sep='\t',
        header=None,
        usecols=[0, 1, 3]  # news_id, category, title
    )
    news_df.columns = ['news_id', 'category', 'title']
    
    # 행동 데이터 로드
    behaviors_path = f"{data_dir}/{split}/behaviors.tsv"
    print(f"   Loading behaviors data: {behaviors_path}")
    behaviors_df = pd.read_csv(
        behaviors_path,
        sep='\t',
        header=None
    )
    behaviors_df.columns = ['impression_id', 'user_id', 'time', 'clicked_news', 'impressions']
    
    print(f"Data loaded: {len(news_df)} news, {len(behaviors_df)} behaviors")
    return news_df, behaviors_df


def build_vocabulary(news_df, save_path):
    """어휘 사전 구축"""
    print("Building vocabulary...")
    
    word_count = defaultdict(int)
    
    # 모든 뉴스 제목에서 단어 수집
    for _, row in tqdm(news_df.iterrows(), total=len(news_df), desc="Processing news titles"):
        if pd.notna(row['title']):
            words = row['title'].lower().split()
            for word in words:
                word_count[word] += 1
    
    # 빈도 기준으로 어휘 구축 (최소 빈도 2 이상)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, count in word_count.items():
        if count >= 2:  # 최소 빈도
            vocab[word] = len(vocab)
    
    # 어휘 저장
    with open(save_path, 'wb') as f:
        pickle.dump(vocab, f)
    
    print(f"Vocabulary built: {len(vocab)} words, saved to {save_path}")
    return vocab


def build_user_news_mappings(behaviors_df, news_df):
    """사용자-뉴스 매핑 구축"""
    print("🔗 Building user-news mappings...")
    
    # 뉴스 정보를 딕셔너리로 변환
    news_dict = {}
    for _, row in news_df.iterrows():
        news_dict[row['news_id']] = {
            'category': row['category'],
            'title': row['title'] if pd.notna(row['title']) else "",
        }
    
    # 사용자 클릭 히스토리 구축
    user_clicked_news = defaultdict(list)
    news_to_users = defaultdict(set)
    user_to_news = defaultdict(set)
    
    print("   Processing user click histories...")
    for _, row in tqdm(behaviors_df.iterrows(), total=len(behaviors_df), desc="Processing behaviors"):
        user_id = row['user_id']
        
        if pd.notna(row['clicked_news']):
            clicked_list = row['clicked_news'].split()
            user_clicked_news[user_id].extend(clicked_list)
            
            # 이분 그래프 매핑 구축
            for news_id in clicked_list:
                if news_id in news_dict:
                    news_to_users[news_id].add(user_id)
                    user_to_news[user_id].add(news_id)
    
    print(f"Mappings built:")
    print(f"   Users: {len(user_to_news)}")
    print(f"   News with clicks: {len(news_to_users)}")
    print(f"   Total user-news interactions: {sum(len(news_set) for news_set in user_to_news.values())}")
    
    return news_dict, user_clicked_news, news_to_users, user_to_news


def build_neighbor_graphs(news_to_users, user_to_news):
    """이웃 그래프 구축 (논문의 핵심 로직)"""
    print("🕸️  Building neighbor graphs...")
    
    # 논문: "동일한 사용자가 본 뉴스로 간주되어 이웃 뉴스로 정의"
    print("   Building news neighbor graph...")
    news_neighbors = defaultdict(set)
    
    for user_id, news_set in tqdm(user_to_news.items(), desc="Processing users for news neighbors"):
        news_list = list(news_set)
        if len(news_list) > 1:  # 최소 2개 이상의 뉴스를 클릭한 사용자만
            for i, news1 in enumerate(news_list):
                for j, news2 in enumerate(news_list):
                    if i != j:
                        news_neighbors[news1].add(news2)
    
    # 논문: "특정 사용자는 동일한 클릭한 뉴스를 공유하여 이웃 사용자로 정의"
    print("   Building user neighbor graph...")
    user_neighbors = defaultdict(set)
    
    for news_id, user_set in tqdm(news_to_users.items(), desc="Processing news for user neighbors"):
        user_list = list(user_set)
        if len(user_list) > 1:  # 최소 2명 이상이 클릭한 뉴스만
            for i, user1 in enumerate(user_list):
                for j, user2 in enumerate(user_list):
                    if i != j:
                        user_neighbors[user1].add(user2)
    
    print(f"Neighbor graphs built:")
    print(f"   News with neighbors: {len(news_neighbors)}")
    print(f"   Users with neighbors: {len(user_neighbors)}")
    
    # 통계 출력
    if news_neighbors:
        avg_news_neighbors = sum(len(neighbors) for neighbors in news_neighbors.values()) / len(news_neighbors)
        max_news_neighbors = max(len(neighbors) for neighbors in news_neighbors.values())
        print(f"   Avg news neighbors: {avg_news_neighbors:.1f}, Max: {max_news_neighbors}")
    
    if user_neighbors:
        avg_user_neighbors = sum(len(neighbors) for neighbors in user_neighbors.values()) / len(user_neighbors)
        max_user_neighbors = max(len(neighbors) for neighbors in user_neighbors.values())
        print(f"   Avg user neighbors: {avg_user_neighbors:.1f}, Max: {max_user_neighbors}")
    
    return news_neighbors, user_neighbors


def save_graph_data(graph_data, save_path):
    """그래프 데이터 저장"""
    print(f"Saving graph data to {save_path}...")
    
    with open(save_path, 'wb') as f:
        pickle.dump(graph_data, f)
    
    # 파일 크기 확인
    file_size = os.path.getsize(save_path) / (1024 * 1024)  # MB
    print(f"Graph saved: {file_size:.1f} MB")


def build_for_split(data_dir, split, vocab=None, force_rebuild=False):
    """특정 split에 대한 그래프 구축"""
    print(f"\nProcessing {split} split...")
    
    # 파일 경로 설정
    vocab_path = os.path.join(data_dir, "vocab.pkl")
    graph_path = os.path.join(data_dir, f"graph_{split}.pkl")
    
    # 기존 파일 확인
    if not force_rebuild and os.path.exists(graph_path):
        print(f"   Graph for {split} already exists: {graph_path}")
        return vocab
    
    start_time = time.time()
    
    # 1. 데이터 로드
    news_df, behaviors_df = load_mind_data(data_dir, split)
    
    # 2. 어휘 구축 (train에서만)
    if split == 'train':
        if force_rebuild or not os.path.exists(vocab_path):
            vocab = build_vocabulary(news_df, vocab_path)
        else:
            print(f"   Loading existing vocabulary from {vocab_path}")
            with open(vocab_path, 'rb') as f:
                vocab = pickle.load(f)
            print(f"   Vocabulary loaded: {len(vocab)} words")
    
    # 3. 사용자-뉴스 매핑 구축
    news_dict, user_clicked_news, news_to_users, user_to_news = build_user_news_mappings(
        behaviors_df, news_df
    )
    
    # 4. 이웃 그래프 구축
    news_neighbors, user_neighbors = build_neighbor_graphs(news_to_users, user_to_news)
    
    # 5. 그래프 데이터 저장
    graph_data = {
        'news_to_users': dict(news_to_users),
        'user_to_news': dict(user_to_news),
        'news_neighbors': dict(news_neighbors),
        'user_neighbors': dict(user_neighbors),
        'news_dict': news_dict,
        'user_clicked_news': dict(user_clicked_news)
    }
    
    save_graph_data(graph_data, graph_path)
    
    # 완료 시간 출력
    elapsed_time = time.time() - start_time
    print(f"   {split.title()} completed in {elapsed_time:.1f}s")
    
    return vocab


def main():
    parser = argparse.ArgumentParser(description="Build bipartite graph for GERL model")
    parser.add_argument('--data_dir', default='data/MIND_small', help='MIND dataset directory')
    parser.add_argument('--split', default=None, choices=['train', 'dev'], 
                       help='Data split to process (default: both train and dev)')
    parser.add_argument('--force_rebuild', action='store_true', help='Force rebuild even if files exist')
    
    args = parser.parse_args()
    
    print(" GERL Graph Builder")
    print("=" * 50)
    print(f"Data directory: {args.data_dir}")
    
    # split 인자가 없으면 둘 다 처리
    if args.split is None:
        splits_to_process = ['train', 'dev']
        print("Processing: train and dev splits")
    else:
        splits_to_process = [args.split]
        print(f"Processing: {args.split} split only")
    
    print("=" * 50)
    
    # 기존 파일 확인 (전체적으로)
    vocab_path = os.path.join(args.data_dir, "vocab.pkl")
    all_exist = os.path.exists(vocab_path)
    for split in splits_to_process:
        graph_path = os.path.join(args.data_dir, f"graph_{split}.pkl")
        all_exist = all_exist and os.path.exists(graph_path)
    
    if not args.force_rebuild and all_exist:
        print(" All graph files already exist!")
        print("   Files found:")
        print(f"     {vocab_path}")
        for split in splits_to_process:
            graph_path = os.path.join(args.data_dir, f"graph_{split}.pkl")
            print(f"     {graph_path}")
        response = input("\nDo you want to rebuild? (y/N): ")
        if response.lower() != 'y':
            print(" Aborted by user")
            return
    
    total_start_time = time.time()
    vocab = None
    
    # train을 먼저 처리 (어휘 구축을 위해)
    if 'train' in splits_to_process:
        vocab = build_for_split(args.data_dir, 'train', vocab, args.force_rebuild)
    
    # dev 처리
    if 'dev' in splits_to_process:
        vocab = build_for_split(args.data_dir, 'dev', vocab, args.force_rebuild)
    
    # 전체 완료 시간 출력
    total_elapsed_time = time.time() - total_start_time
    print("\n" + "=" * 50)
    print("All graph building completed!")
    print(f" Total time: {total_elapsed_time:.1f} seconds ({total_elapsed_time/60:.1f} minutes)")
    print("=" * 50)
    
    print("\nSummary:")
    print(f"  Data directory: {args.data_dir}")
    if vocab:
        print(f"  Vocabulary: {len(vocab)} words")
    print(f"  Files created:")
    print(f"     {vocab_path}")
    for split in splits_to_process:
        graph_path = os.path.join(args.data_dir, f"graph_{split}.pkl")
        print(f"     {graph_path}")
    
    print("\n Now you can run training:")
    print("   python train.py --epochs 5 --batch_size 32")


if __name__ == "__main__":
    main() 