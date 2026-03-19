"""
모델 베이스 클래스
모든 분류 모델이 공통으로 구현해야 하는 인터페이스를 정의합니다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class PredictResult:
    label: str           # "spam" | "ham"
    confidence: float    # 예측된 클래스의 확률
    spam_probability: float
    ham_probability: float


class BaseSpamClassifier(ABC):
    """스팸 분류기 추상 베이스 클래스."""

    SPAM_THRESHOLD: float = 0.7

    @abstractmethod
    def fit(self, X_train: list[str], y_train: list[int],
            X_val: list[str] | None = None, y_val: list[int] | None = None) -> None:
        """모델을 학습합니다."""

    @abstractmethod
    def predict_proba(self, texts: list[str]) -> list[tuple[float, float]]:
        """
        각 텍스트에 대한 (ham_prob, spam_prob) 튜플 리스트를 반환합니다.
        """

    def predict(self, texts: list[str]) -> list[PredictResult]:
        """texts 리스트에 대해 PredictResult 리스트를 반환합니다."""
        probas = self.predict_proba(texts)
        results = []
        for ham_prob, spam_prob in probas:
            if spam_prob >= self.SPAM_THRESHOLD:
                label = "spam"
                confidence = spam_prob
            else:
                label = "ham"
                confidence = ham_prob
            results.append(PredictResult(label, confidence, spam_prob, ham_prob))
        return results

    def predict_single(self, text: str) -> PredictResult:
        """단일 텍스트를 분류합니다."""
        return self.predict([text])[0]

    @abstractmethod
    def save(self, path: str) -> None:
        """모델을 파일로 저장합니다."""

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseSpamClassifier":
        """파일에서 모델을 불러옵니다."""
