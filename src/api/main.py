"""
FastAPI 서버
스팸 분류 REST API를 제공합니다.

실행:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException

from src.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)
from src.data.preprocessor import Preprocessor

# ------------------------------------------------------------------
# 전역 상태
# ------------------------------------------------------------------

_models: dict = {}
_preprocessor: Preprocessor | None = None
_cfg: dict = {}


def _load_config(path: str = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 모델 로드."""
    global _preprocessor, _cfg
    _cfg = _load_config()
    pre_cfg = _cfg["preprocessing"]

    _preprocessor = Preprocessor(
        use_konlpy=pre_cfg["use_konlpy"],
        konlpy_analyzer=pre_cfg["konlpy_analyzer"],
    )

    # ML 모델
    ml_path = _cfg["artifacts"]["ml_model"]
    if Path(ml_path).exists():
        from src.models.ml_model import MLSpamClassifier
        _models["ml"] = MLSpamClassifier.load(ml_path)
        print(f"[API] ML 모델 로드 완료: {ml_path}")
    else:
        print(f"[API] ML 모델 없음 (학습 후 사용 가능): {ml_path}")

    # BERT 모델
    bert_path = _cfg["artifacts"]["bert_model"]
    if Path(bert_path).exists():
        from src.models.bert_model import BERTSpamClassifier
        _models["bert"] = BERTSpamClassifier.load(bert_path)
        print(f"[API] BERT 모델 로드 완료: {bert_path}")
    else:
        print(f"[API] BERT 모델 없음 (학습 후 사용 가능): {bert_path}")

    yield


app = FastAPI(
    title="Korean Spam Email Classifier",
    description="한국어 스팸 메일 분류 API",
    version="1.0.0",
    lifespan=lifespan,
)


# ------------------------------------------------------------------
# 엔드포인트
# ------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        ml_model_loaded="ml" in _models,
        bert_model_loaded="bert" in _models,
    )


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(
        ml_model=_models["ml"].model_name if "ml" in _models else None,
        bert_model=_models["bert"].model_name if "bert" in _models else None,
        spam_threshold=_cfg.get("ml_model", {}).get("spam_threshold", 0.7),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    model_key = req.model
    if model_key not in _models:
        raise HTTPException(
            status_code=503,
            detail=f"'{model_key}' 모델이 로드되지 않았습니다. 먼저 학습을 실행하세요.",
        )

    model = _models[model_key]
    text = _preprocessor.preprocess(req.subject, req.body)

    t0 = time.perf_counter()
    result = model.predict_single(text)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return PredictResponse(
        label=result.label,
        confidence=result.confidence,
        spam_probability=result.spam_probability,
        ham_probability=result.ham_probability,
        model=model_key,
        processing_time_ms=round(elapsed_ms, 2),
    )


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest) -> BatchPredictResponse:
    # 모델별로 그룹화하여 배치 추론
    from collections import defaultdict

    groups: dict[str, list[int]] = defaultdict(list)
    for i, email in enumerate(req.emails):
        groups[email.model].append(i)

    # 처리 전에 필요한 모델이 모두 로드됐는지 먼저 검증
    for model_key in groups:
        if model_key not in _models:
            raise HTTPException(
                status_code=503,
                detail=f"'{model_key}' 모델이 로드되지 않았습니다. 먼저 학습을 실행하세요.",
            )

    responses: list[PredictResponse | None] = [None] * len(req.emails)

    t0 = time.perf_counter()

    for model_key, indices in groups.items():
        model = _models[model_key]
        texts = [
            _preprocessor.preprocess(req.emails[i].subject, req.emails[i].body)
            for i in indices
        ]
        results = model.predict(texts)

        for i, result in zip(indices, results):
            responses[i] = PredictResponse(
                label=result.label,
                confidence=result.confidence,
                spam_probability=result.spam_probability,
                ham_probability=result.ham_probability,
                model=model_key,
                processing_time_ms=0.0,
            )

    total_ms = (time.perf_counter() - t0) * 1000

    return BatchPredictResponse(
        results=responses,
        total=len(responses),
        total_processing_time_ms=round(total_ms, 2),
    )
