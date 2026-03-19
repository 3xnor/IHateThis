"""
학습 파이프라인
Usage:
    python -m src.train --model ml          # TF-IDF + LR 학습
    python -m src.train --model bert        # KR-ELECTRA 파인튜닝
    python -m src.train --model ml bert     # 둘 다 학습
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.data.loader import load_dataset
from src.data.preprocessor import Preprocessor


def load_config(path: str = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_ml(cfg: dict, data) -> None:
    from src.models.ml_model import MLSpamClassifier

    ml_cfg = cfg["ml_model"]
    tfidf_cfg = ml_cfg["tfidf"]
    lr_cfg = ml_cfg["logistic_regression"]

    model = MLSpamClassifier(
        max_features=tfidf_cfg["max_features"],
        ngram_range=tuple(tfidf_cfg["ngram_range"]),
        sublinear_tf=tfidf_cfg["sublinear_tf"],
        min_df=tfidf_cfg.get("min_df", 2),
        C=lr_cfg["C"],
        max_iter=lr_cfg["max_iter"],
        class_weight=lr_cfg["class_weight"],
        solver=lr_cfg.get("solver", "lbfgs"),
        spam_threshold=ml_cfg["spam_threshold"],
    )

    model.fit(data.X_train, data.y_train, data.X_val, data.y_val)
    model.save(cfg["artifacts"]["ml_model"])
    return model


def train_bert(cfg: dict, data) -> None:
    from src.models.bert_model import BERTSpamClassifier

    bert_cfg = cfg["bert_model"]
    model = BERTSpamClassifier(
        base_model=bert_cfg["base_model"],
        max_length=bert_cfg["max_length"],
        batch_size=bert_cfg["batch_size"],
        epochs=bert_cfg["epochs"],
        learning_rate=bert_cfg["learning_rate"],
        warmup_ratio=bert_cfg["warmup_ratio"],
        weight_decay=bert_cfg["weight_decay"],
        spam_threshold=bert_cfg["spam_threshold"],
    )

    model.fit(data.X_train, data.y_train, data.X_val, data.y_val)
    model.save(cfg["artifacts"]["bert_model"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="스팸 분류 모델 학습")
    parser.add_argument(
        "--model",
        nargs="+",
        choices=["ml", "bert"],
        default=["ml"],
        help="학습할 모델 (ml | bert | ml bert)",
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

    Path(cfg["artifacts"]["dir"]).mkdir(parents=True, exist_ok=True)

    if "ml" in args.model:
        train_ml(cfg, data)

    if "bert" in args.model:
        train_bert(cfg, data)

    print("\n[완료] 학습이 끝났습니다.")


if __name__ == "__main__":
    main()
