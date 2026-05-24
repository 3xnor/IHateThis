"""
Gmail 연동 라우터 (UC-06)

엔드포인트:
  GET  /gmail/connect      - Google OAuth URL 반환
  GET  /gmail/callback     - 인증 코드 → 토큰 교환 및 저장
  POST /gmail/classify     - Gmail 메일 가져와서 스팸 분류
  DELETE /gmail/disconnect - 토큰 삭제 및 Google 측 폐기
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from email import message_from_bytes
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from pydantic import BaseModel

router = APIRouter(prefix="/gmail", tags=["Gmail"])

# 토큰 저장 파일 경로 (.gitignore에 포함됨)
TOKEN_PATH = Path("gmail_token.json")

# connect → callback 사이에 Flow 객체 보관 (PKCE code_verifier 유지용)
_flow_store: dict[str, Flow] = {}

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

CLIENT_CONFIG = {
    "web": {
        "client_id": os.getenv("GOOGLE_CLIENT_ID"),
        "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
        "redirect_uris": [os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/gmail/callback")],
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
    }
}


# ------------------------------------------------------------------
# 스키마
# ------------------------------------------------------------------

class GmailClassifyRequest(BaseModel):
    max_results: int = 10
    label_filter: str = "INBOX"
    model: str = "ml"


class GmailClassifyResult(BaseModel):
    gmail_id: str
    subject: str
    sender: str
    date: str
    label: str
    confidence: float
    spam_probability: float


# ------------------------------------------------------------------
# 토큰 저장 / 로드
# ------------------------------------------------------------------

def _save_token(creds: Credentials) -> None:
    TOKEN_PATH.write_text(
        json.dumps({
            "token": creds.token,
            "refresh_token": creds.refresh_token,
            "token_uri": creds.token_uri,
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
            "scopes": list(creds.scopes or SCOPES),
        }),
        encoding="utf-8",
    )


def _load_token() -> Credentials | None:
    if not TOKEN_PATH.exists():
        return None
    data = json.loads(TOKEN_PATH.read_text(encoding="utf-8"))
    return Credentials(
        token=data["token"],
        refresh_token=data.get("refresh_token"),
        token_uri=data["token_uri"],
        client_id=data["client_id"],
        client_secret=data["client_secret"],
        scopes=data["scopes"],
    )


def _get_valid_credentials() -> Credentials:
    creds = _load_token()
    if creds is None:
        raise HTTPException(status_code=401, detail="Gmail 연동이 필요합니다. GET /gmail/connect 를 먼저 호출하세요.")
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        _save_token(creds)
    return creds


# ------------------------------------------------------------------
# 이메일 파싱 유틸
# ------------------------------------------------------------------

def _decode_body(payload: dict) -> str:
    """Gmail API payload에서 본문 텍스트 추출."""
    if "parts" in payload:
        for part in payload["parts"]:
            if part.get("mimeType") == "text/plain":
                data = part["body"].get("data", "")
                if data:
                    return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="ignore")
        # text/plain 없으면 text/html에서 태그 제거
        for part in payload["parts"]:
            if part.get("mimeType") == "text/html":
                data = part["body"].get("data", "")
                if data:
                    html = base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="ignore")
                    return re.sub(r"<[^>]+>", " ", html)
    # 단일 파트
    data = payload.get("body", {}).get("data", "")
    if data:
        return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="ignore")
    return ""


def _get_header(headers: list[dict], name: str) -> str:
    for h in headers:
        if h["name"].lower() == name.lower():
            return h["value"]
    return ""


# ------------------------------------------------------------------
# 엔드포인트
# ------------------------------------------------------------------

@router.get("/connect")
def gmail_connect():
    """Google OAuth 2.0 인증 URL을 반환합니다."""
    flow = Flow.from_client_config(CLIENT_CONFIG, scopes=SCOPES)
    flow.redirect_uri = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/gmail/callback")
    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    # PKCE code_verifier를 유지하기 위해 Flow 객체를 state 키로 보관
    _flow_store[state] = flow
    return {"auth_url": auth_url}


@router.get("/callback")
def gmail_callback(code: str, state: str):
    """Google가 전달한 인증 코드를 토큰으로 교환하고 저장합니다."""
    flow = _flow_store.pop(state, None)
    if flow is None:
        raise HTTPException(status_code=400, detail="인증 세션이 만료됐거나 잘못된 요청입니다. /gmail/connect 부터 다시 시작하세요.")
    flow.fetch_token(code=code)
    _save_token(flow.credentials)
    return {"message": "Gmail 연동이 완료되었습니다."}


@router.post("/classify", response_model=list[GmailClassifyResult])
def gmail_classify(req: GmailClassifyRequest):
    """Gmail에서 메일을 가져와 스팸 분류를 수행합니다."""
    from src.api.main import _models, _preprocessor

    if req.model not in _models:
        raise HTTPException(status_code=503, detail=f"'{req.model}' 모델이 로드되지 않았습니다.")

    creds = _get_valid_credentials()
    service = build("gmail", "v1", credentials=creds)

    # 메일 목록 조회
    list_result = service.users().messages().list(
        userId="me",
        labelIds=[req.label_filter],
        maxResults=req.max_results,
    ).execute()

    messages = list_result.get("messages", [])
    if not messages:
        return []

    results: list[GmailClassifyResult] = []
    model = _models[req.model]

    for msg_ref in messages:
        msg = service.users().messages().get(
            userId="me", id=msg_ref["id"], format="full"
        ).execute()

        headers = msg["payload"].get("headers", [])
        subject = _get_header(headers, "Subject")
        sender = _get_header(headers, "From")
        date = _get_header(headers, "Date")
        body = _decode_body(msg["payload"])

        text = _preprocessor.preprocess(subject, body)
        prediction = model.predict_single(text)

        results.append(GmailClassifyResult(
            gmail_id=msg_ref["id"],
            subject=subject,
            sender=sender,
            date=date,
            label=prediction.label,
            confidence=round(prediction.confidence, 4),
            spam_probability=round(prediction.spam_probability, 4),
        ))

    return results


@router.delete("/disconnect")
def gmail_disconnect():
    """저장된 OAuth 토큰을 삭제합니다."""
    import requests as req_lib

    creds = _load_token()
    if creds is None:
        raise HTTPException(status_code=404, detail="연동된 Gmail 계정이 없습니다.")

    # Google 측 토큰 폐기
    if creds.token:
        req_lib.post(
            "https://oauth2.googleapis.com/revoke",
            params={"token": creds.token},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

    TOKEN_PATH.unlink(missing_ok=True)
    return {"message": "Gmail 연동이 해제되었습니다."}
