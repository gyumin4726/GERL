import pandas as pd

def check_data_structure():
    # 첫 번째 줄 읽기 - 컬럼 수 확인
    print("news.tsv 첫 번째 줄 확인:")
    with open("data/MIND_small/train/news.tsv", 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        columns = first_line.split('\t')
        print(f"컬럼 수: {len(columns)}")
        print("컬럼들:", columns[:5], "..." if len(columns) > 5 else "")
    
    print("\nbehaviors.tsv 첫 번째 줄 확인:")
    with open("data/MIND_small/train/behaviors.tsv", 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        columns = first_line.split('\t')
        print(f"컬럼 수: {len(columns)}")
        print("컬럼들:", columns)
    
    # 실제 로딩 테스트
    print("\n실제 로딩 테스트:")
    try:
        news_df = pd.read_csv(
            "data/MIND_small/train/news.tsv",
            sep='\t',
            header=None,
            nrows=5  # 처음 5줄만
        )
        print(f"news.tsv shape: {news_df.shape}")
        print("news.tsv 컬럼들:")
        for i, col in enumerate(news_df.columns):
            print(f"  {i}: {news_df.iloc[0, i]}")
    except Exception as e:
        print(f"news.tsv 로딩 실패: {e}")
    
    try:
        behaviors_df = pd.read_csv(
            "data/MIND_small/train/behaviors.tsv",
            sep='\t',
            header=None,
            nrows=3  # 처음 3줄만
        )
        print(f"\nbehaviors.tsv shape: {behaviors_df.shape}")
        print("behaviors.tsv 컬럼들:")
        for i, col in enumerate(behaviors_df.columns):
            print(f"  {i}: {behaviors_df.iloc[0, i]}")
    except Exception as e:
        print(f"behaviors.tsv 로딩 실패: {e}")

if __name__ == "__main__":
    check_data_structure() 