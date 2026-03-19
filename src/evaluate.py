"""
평가 스크립트
학습된 모델을 test 세트로 평가하고 리포트를 출력합니다.

Usage:
    python -m src.evaluate --model ml
    python -m src.evaluate --model bert
"""

from __future__ import annotations

import argparse
import time

import yaml
from sklearn.metrics import classification_report, confusion_matrix

from src.data.loader import load_dataset
from src.data.preprocessor import Preprocessor


def load_config(path: str = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate(model, X_test: list[str], y_test: list[int], model_name: str) -> None:
    print(f"\n{'='*50}")
    print(f"모델: {model_name}")
    print(f"{'='*50}")

    start = time.perf_counter()
    results = model.predict(X_test)
    elapsed_ms = (time.perf_counter() - start) * 1000

    y_pred = [1 if r.label == "spam" else 0 for r in results]

    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))
    print("혼동 행렬:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\n추론 시간: {elapsed_ms:.1f}ms ({len(X_test)}개 | "
          f"평균 {elapsed_ms/len(X_test):.2f}ms/건)")


def main() -> None:
    parser = argparse.ArgumentParser(description="스팸 분류 모델 평가")
    parser.add_argument(
        "--model",
        choices=["ml", "bert"],
        default="ml",
        help="평가할 모델 (ml | bert)",
    )
    parser.add_argument("--config", default="config.yaml", help="설정 파일 경로")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pre_cfg = cfg["preprocessing"]

    preprocessor = Preprocessor(
        use_konlpy=pre_cfg["use_konlpy"],
        konlpy_analyzer=pre_cfg["konlpy_analyzer"],
    )

    data = load_dataset(
        csv_path=cfg["data"]["path"],
        preprocessor=preprocessor,
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
        random_state=cfg["data"]["random_state"],
    )

    if args.model == "ml":
        from src.models.ml_model import MLSpamClassifier
        model = MLSpamClassifier.load(cfg["artifacts"]["ml_model"])
        evaluate(model, data.X_test, data.y_test, "TF-IDF + Logistic Regression")

    elif args.model == "bert":
        from src.models.bert_model import BERTSpamClassifier
        model = BERTSpamClassifier.load(cfg["artifacts"]["bert_model"])
        evaluate(model, data.X_test, data.y_test, "KR-ELECTRA")


if __name__ == "__main__":
    main()
