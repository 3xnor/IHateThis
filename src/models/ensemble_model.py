"""
앙상블 모델 — ML + BERT 가중 평균
두 모델의 스팸 확률을 가중 평균하여 단일 모델보다 안정적인 예측을 제공합니다.
"""

from __future__ import annotations

from src.models.base import BaseSpamClassifier, PredictResult


class EnsembleClassifier(BaseSpamClassifier):
    """
    MLSpamClassifier + BERTSpamClassifier 가중 평균 앙상블.

    spam_prob = ml_weight * ml_spam_prob + bert_weight * bert_spam_prob

    Parameters
    ----------
    ml_model : BaseSpamClassifier
        학습된 ML 모델
    bert_model : BaseSpamClassifier
        학습된 BERT 모델
    ml_weight : float
        ML 모델 가중치 (기본 0.3)
    bert_weight : float
        BERT 모델 가중치 (기본 0.7)
    spam_threshold : float
        앙상블 스팸 판정 임계값 (기본 0.7)
    """

    def __init__(
        self,
        ml_model: BaseSpamClassifier,
        bert_model: BaseSpamClassifier,
        ml_weight: float = 0.3,
        bert_weight: float = 0.7,
        spam_threshold: float = 0.7,
    ) -> None:
        if abs(ml_weight + bert_weight - 1.0) > 1e-6:
            raise ValueError("ml_weight + bert_weight는 1.0이어야 합니다.")
        self.ml_model = ml_model
        self.bert_model = bert_model
        self.ml_weight = ml_weight
        self.bert_weight = bert_weight
        self.SPAM_THRESHOLD = spam_threshold

    # ------------------------------------------------------------------
    # BaseSpamClassifier 구현
    # ------------------------------------------------------------------

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> None:
        raise NotImplementedError(
            "EnsembleClassifier는 직접 학습하지 않습니다. "
            "ML 모델과 BERT 모델을 각각 학습한 후 조합하세요."
        )

    def predict_proba(self, texts: list[str]) -> list[tuple[float, float]]:
        ml_probas   = self.ml_model.predict_proba(texts)
        bert_probas = self.bert_model.predict_proba(texts)

        results = []
        for (ml_ham, ml_spam), (bert_ham, bert_spam) in zip(ml_probas, bert_probas):
            spam = self.ml_weight * ml_spam + self.bert_weight * bert_spam
            ham  = 1.0 - spam
            results.append((ham, spam))
        return results

    def save(self, path: str) -> None:
        raise NotImplementedError("앙상블 모델은 개별 모델을 저장하세요.")

    @classmethod
    def load(cls, path: str) -> "EnsembleClassifier":
        raise NotImplementedError("앙상블 모델은 개별 모델을 로드한 후 조합하세요.")

    @property
    def model_name(self) -> str:
        return f"Ensemble (ML×{self.ml_weight} + BERT×{self.bert_weight})"
