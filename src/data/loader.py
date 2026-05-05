"""
데이터 로더
CSV 파일을 읽고 train/validation/test 세트로 분리합니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.preprocessor import Preprocessor


@dataclass
class DataSplit:
    X_train: list[str]
    X_val: list[str]
    X_test: list[str]
    y_train: list[int]
    y_val: list[int]
    y_test: list[int]


def load_datasets(
    csv_paths: list[str | Path],
    preprocessor: Preprocessor,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    random_state: int = 42,
) -> DataSplit:
    """여러 CSV를 합쳐서 train/val/test로 분리합니다."""
    dfs = []
    for p in csv_paths:
        df_part = pd.read_csv(p, encoding="utf-8-sig")
        _validate_columns(df_part)
        dfs.append(df_part)
    merged = pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
    print(f"[데이터] {len(csv_paths)}개 CSV 병합 → 총 {len(merged)}개")
    return _split_df(merged, preprocessor, train_ratio, val_ratio, random_state)


def load_dataset(
    csv_path: str | Path,
    preprocessor: Preprocessor,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    random_state: int = 42,
) -> DataSplit:
    """
    CSV를 로드하고 전처리 후 train/val/test로 분리합니다.

    Parameters
    ----------
    csv_path : str | Path
        데이터셋 경로 (data/korean_spam_dataset.csv)
    preprocessor : Preprocessor
        전처리 객체
    train_ratio : float
        학습 비율 (기본 0.8)
    val_ratio : float
        검증 비율 (기본 0.1, 나머지는 test)
    random_state : int
        재현을 위한 시드
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    _validate_columns(df)
    print(f"[데이터] 총 {len(df)}개 로드")
    return _split_df(df, preprocessor, train_ratio, val_ratio, random_state)


def _split_df(
    df: pd.DataFrame,
    preprocessor: Preprocessor,
    train_ratio: float,
    val_ratio: float,
    random_state: int,
) -> DataSplit:
    label_col = df["label"].astype(str)
    spam_count = int((label_col == "spam").sum())
    ham_count  = int((label_col == "ham").sum())
    print(f"  스팸: {spam_count}개 | 정상: {ham_count}개")

    # 전처리
    print("[전처리] 텍스트 정제 중...")
    texts = [
        preprocessor.preprocess(row["subject"], row["body"])
        for _, row in df.iterrows()
    ]
    _label_map = {"spam": 1, "ham": 0}
    labels = [_label_map[str(l)] for l in df["label"]]

    # train / (val + test) 분리
    test_ratio = 1.0 - train_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels,
        test_size=test_ratio,
        random_state=random_state,
        stratify=labels,
    )

    # val / test 분리
    val_frac = val_ratio / test_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=1.0 - val_frac,
        random_state=random_state,
        stratify=y_temp,
    )

    print(f"[분리] train={len(X_train)} | val={len(X_val)} | test={len(X_test)}")
    return DataSplit(X_train, X_val, X_test, y_train, y_val, y_test)


def _validate_columns(df: pd.DataFrame) -> None:
    required = {"id", "subject", "body", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV에 필수 컬럼이 없습니다: {missing}")
