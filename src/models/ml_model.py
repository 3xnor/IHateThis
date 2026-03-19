"""
경량 ML 모델 — TF-IDF + Logistic Regression
GPU 없이도 빠르게 추론 가능한 베이스라인 모델입니다.
"""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.models.base import BaseSpamClassifier, PredictResult


class MLSpamClassifier(BaseSpamClassifier):
    """
    TF-IDF 벡터화 + Logistic Regression 파이프라인.

    설정값은 config.yaml의 ml_model 섹션을 따릅니다.
    """

    def __init__(
        self,
        max_features: int = 50_000,
        ngram_range: tuple[int, int] = (1, 2),
        sublinear_tf: bool = True,
        min_df: int = 2,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str = "balanced",
        solver: str = "lbfgs",
        spam_threshold: float = 0.7,
    ) -> None:
        self.SPAM_THRESHOLD = spam_threshold
        self._pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=max_features,
                        ngram_range=ngram_range,
                        sublinear_tf=sublinear_tf,
                        min_df=min_df,
                        analyzer="char_wb",  # 한국어 형태소 미분리 시 문자 n-gram 사용
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=C,
                        max_iter=max_iter,
                        class_weight=class_weight,
                        solver=solver,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    # ------------------------------------------------------------------
    # BaseSpamClassifier 구현
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: list[str],
        y_train: list[int],
        X_val: list[str] | None = None,
        y_val: list[int] | None = None,
    ) -> None:
        print("[ML] 학습 시작...")
        self._pipeline.fit(X_train, y_train)
        print("[ML] 학습 완료")

        if X_val is not None and y_val is not None:
            from sklearn.metrics import classification_report
            y_pred = self._pipeline.predict(X_val)
            print("[ML] Validation 결과:")
            print(classification_report(y_val, y_pred, target_names=["ham", "spam"]))

    def predict_proba(self, texts: list[str]) -> list[tuple[float, float]]:
        probas = self._pipeline.predict_proba(texts)
        # scikit-learn은 클래스 순서가 [0(ham), 1(spam)]
        return [(float(p[0]), float(p[1])) for p in probas]

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, path)
        print(f"[ML] 모델 저장: {path}")

    @classmethod
    def load(cls, path: str) -> "MLSpamClassifier":
        obj = cls.__new__(cls)
        obj._pipeline = joblib.load(path)
        obj.SPAM_THRESHOLD = 0.7
        print(f"[ML] 모델 로드: {path}")
        return obj

    # ------------------------------------------------------------------
    # 편의 메서드
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return "TF-IDF + Logistic Regression"
