"""
하이퍼파라미터 튜닝
Usage:
    python -m src.tune --model ml          # RandomizedSearchCV로 ML 모델 튜닝
    python -m src.tune --model bert        # Optuna로 BERT 모델 튜닝
    python -m src.tune --model ml bert     # 둘 다 튜닝
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────
# ML 튜닝 — RandomizedSearchCV
# ─────────────────────────────────────────────────────────────

def tune_ml(cfg: dict, data) -> dict:
    """
    TF-IDF + LR 파이프라인의 하이퍼파라미터를 RandomizedSearchCV로 탐색합니다.
    최적 파라미터를 반환하고 config.yaml을 업데이트합니다.
    """
    from scipy.stats import loguniform, randint
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.pipeline import Pipeline

    print("\n[ML 튜닝] RandomizedSearchCV 시작...")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            sublinear_tf=True,
            class_weight=None,
            analyzer="char_wb",
        )),
        ("clf", LogisticRegression(
            class_weight="balanced",
            solver="lbfgs",
            n_jobs=-1,
            max_iter=1000,
        )),
    ])

    param_dist = {
        "tfidf__max_features": [10_000, 30_000, 50_000, 80_000],
        "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3), (2, 3)],
        "tfidf__min_df": [1, 2, 3, 5],
        "clf__C": loguniform(1e-2, 1e2),
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=20,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        verbose=2,
        random_state=42,
    )

    # train + val을 합쳐서 CV 탐색
    X_search = data.X_train + data.X_val
    y_search = data.y_train + data.y_val

    search.fit(X_search, y_search)

    best = search.best_params_
    print(f"\n[ML 튜닝] 최적 파라미터:")
    for k, v in best.items():
        print(f"  {k}: {v}")
    print(f"  CV F1: {search.best_score_:.4f}")

    # config.yaml 업데이트
    cfg["ml_model"]["tfidf"]["max_features"] = best["tfidf__max_features"]
    cfg["ml_model"]["tfidf"]["ngram_range"] = list(best["tfidf__ngram_range"])
    cfg["ml_model"]["tfidf"]["min_df"] = best["tfidf__min_df"]
    cfg["ml_model"]["logistic_regression"]["C"] = round(float(best["clf__C"]), 6)
    _save_config(cfg)

    print("[ML 튜닝] config.yaml 업데이트 완료")
    return best


# ─────────────────────────────────────────────────────────────
# BERT 튜닝 — Optuna
# ─────────────────────────────────────────────────────────────

def tune_bert(cfg: dict, data) -> dict:
    """
    KR-ELECTRA 파인튜닝 하이퍼파라미터를 Optuna로 탐색합니다.
    각 trial은 2 에폭만 학습하여 빠르게 탐색합니다.
    최적 파라미터를 반환하고 config.yaml을 업데이트합니다.
    """
    import optuna
    from src.models.bert_model import BERTSpamClassifier

    bert_cfg = cfg["bert_model"]
    N_TRIALS = 10       # 탐색 횟수 (늘릴수록 정확하지만 느림)
    TUNE_EPOCHS = 2     # 튜닝 중 사용할 에폭 수 (빠른 탐색용)

    print(f"\n[BERT 튜닝] Optuna 시작 (trials={N_TRIALS}, epochs={TUNE_EPOCHS})...")

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
        warmup = trial.suggest_float("warmup_ratio", 0.05, 0.2)
        wd = trial.suggest_float("weight_decay", 0.0, 0.1)
        bs = trial.suggest_categorical("batch_size", [16, 32])

        model = BERTSpamClassifier(
            base_model=bert_cfg["base_model"],
            max_length=bert_cfg["max_length"],
            batch_size=bs,
            epochs=TUNE_EPOCHS,
            learning_rate=lr,
            warmup_ratio=warmup,
            weight_decay=wd,
            spam_threshold=bert_cfg["spam_threshold"],
        )
        model.fit(data.X_train, data.y_train, data.X_val, data.y_val)

        from sklearn.metrics import f1_score
        preds = [r.label for r in model.predict(data.X_val)]
        y_pred = [1 if p == "spam" else 0 for p in preds]
        return f1_score(data.y_val, y_pred, pos_label=1)

    # 로그 레벨 낮춰서 출력 정리
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_params
    print(f"\n[BERT 튜닝] 최적 파라미터:")
    for k, v in best.items():
        print(f"  {k}: {v}")
    print(f"  Val F1: {study.best_value:.4f}")

    # config.yaml 업데이트
    cfg["bert_model"]["learning_rate"] = float(best["learning_rate"])
    cfg["bert_model"]["warmup_ratio"] = float(best["warmup_ratio"])
    cfg["bert_model"]["weight_decay"] = float(best["weight_decay"])
    cfg["bert_model"]["batch_size"] = int(best["batch_size"])
    _save_config(cfg)

    print("[BERT 튜닝] config.yaml 업데이트 완료")
    return best


# ─────────────────────────────────────────────────────────────
# 공통
# ─────────────────────────────────────────────────────────────

def _save_config(cfg: dict, path: str = "config.yaml") -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="하이퍼파라미터 튜닝")
    parser.add_argument(
        "--model",
        nargs="+",
        choices=["ml", "bert"],
        default=["ml"],
        help="튜닝할 모델 (ml | bert | ml bert)",
    )
    parser.add_argument("--config", default="config.yaml", help="설정 파일 경로")
    args = parser.parse_args()

    cfg = load_config(args.config)

    from src.data.loader import load_dataset
    from src.data.preprocessor import Preprocessor

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
        tune_ml(cfg, data)

    if "bert" in args.model:
        tune_bert(cfg, data)

    print("\n[완료] 튜닝이 끝났습니다. config.yaml에 최적 파라미터가 저장되었습니다.")
    print("이제 python -m src.train 으로 최적 파라미터로 학습하세요.")


if __name__ == "__main__":
    main()
