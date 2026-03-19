"""
API 요청/응답 스키마
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    subject: str = Field(..., description="이메일 제목")
    body: str = Field(..., description="이메일 본문")
    model: Literal["ml", "bert"] = Field("bert", description="사용할 모델 (ml | bert)")


class PredictResponse(BaseModel):
    label: Literal["spam", "ham"] = Field(..., description="분류 결과")
    confidence: float = Field(..., description="예측 클래스의 확률", ge=0.0, le=1.0)
    spam_probability: float = Field(..., description="스팸일 확률", ge=0.0, le=1.0)
    ham_probability: float = Field(..., description="정상일 확률", ge=0.0, le=1.0)
    model: str = Field(..., description="사용된 모델 이름")
    processing_time_ms: float = Field(..., description="처리 시간 (ms)")


class BatchPredictRequest(BaseModel):
    emails: list[PredictRequest] = Field(..., description="분류할 이메일 목록", max_length=100)


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]
    total: int
    total_processing_time_ms: float


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    ml_model_loaded: bool
    bert_model_loaded: bool


class ModelInfoResponse(BaseModel):
    ml_model: str | None
    bert_model: str | None
    spam_threshold: float
