# 스팸 메일 분류 AI

이메일을 분석하여 스팸(spam) / 정상(ham)을 자동으로 분류하는 머신러닝 기반 시스템입니다.

---

## 개발 현황

| 단계 | 내용 | 상태 |
|------|------|------|
| 1단계 | 한국어 데이터셋 생성 | 완료 |
| 2단계 | 전처리 파이프라인 구축 | 완료 |
| 3단계 | TF-IDF + LR 베이스라인 모델 | 완료 |
| 4단계 | BERT 모델 파인튜닝 | 완료 |
| 5단계 | FastAPI 서버 구축 | 완료 |

---

## 주요 기능

- 이메일 텍스트(제목 + 본문)를 입력받아 스팸 여부 판단
- 스팸일 확률(confidence score) 함께 반환
- REST API 서버 제공 (FastAPI)
- 두 가지 모델 지원: 경량 ML 모델 / 고성능 BERT 기반 모델

---

## 기술 스택

| 구분 | 기술 |
|------|------|
| 언어 | Python 3.10+ |
| 경량 모델 | scikit-learn (TF-IDF + Logistic Regression) |
| 고성능 모델 | KR-ELECTRA (HuggingFace Transformers) |
| API 서버 | FastAPI + Uvicorn |
| 데이터 처리 | pandas, numpy |
| 평가 지표 | scikit-learn metrics |

---

## 프로젝트 구조

```
IHateThis/
├── data/
│   └── korean_spam_dataset.csv   # 5,000개 한국어 이메일
├── notebooks/                    # 단계별 실습 노트북
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_ml_model.ipynb
│   ├── 04_bert_model.ipynb
│   └── 05_api_demo.ipynb
├── src/
│   ├── data/
│   │   ├── loader.py             # 데이터 로딩 & 분리
│   │   └── preprocessor.py       # 텍스트 전처리
│   ├── models/
│   │   ├── base.py               # 모델 추상 인터페이스
│   │   ├── ml_model.py           # TF-IDF + LR 모델
│   │   └── bert_model.py         # KR-ELECTRA 모델
│   ├── api/
│   │   ├── main.py               # FastAPI 앱
│   │   └── schemas.py            # 요청/응답 스키마
│   ├── train.py                  # 학습 실행 스크립트
│   └── evaluate.py               # 평가 스크립트
├── artifacts/                    # 학습된 모델 저장 위치
├── generate_dataset.py           # 데이터셋 생성 스크립트
├── config.yaml                   # 모델 및 학습 설정
├── requirements.txt              # 의존성 목록
├── DESIGN.md
└── README.md
```

---

## 빠른 시작

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. ML 모델 학습 (GPU 불필요, 수 초)
python -m src.train --model ml

# 3. BERT 모델 학습 (GPU 권장, 수십 분)
python -m src.train --model bert

# 4. 평가
python -m src.evaluate --model ml

# 5. API 서버 실행 (모델 학습 후)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

단계별 설명은 `notebooks/` 폴더의 Jupyter 노트북을 참고하세요.

---

## 데이터셋

`generate_dataset.py`로 생성한 한국어 합성 데이터셋입니다.

```bash
python generate_dataset.py   # data/korean_spam_dataset.csv 재생성
```

### 구성

| 항목 | 내용 |
|------|------|
| 전체 | 5,000개 |
| 스팸 | 2,000개 (40%) |
| 정상 | 3,000개 (60%) |
| 컬럼 | id, subject, body, label, category |
| 인코딩 | UTF-8 (BOM) |

### 카테고리

**스팸 (7종)**

| 카테고리 | 예시 제목 |
|---------|---------|
| 대출/금융 | 신용등급 상관없이 300만원 대출 승인! |
| 당첨/이벤트 | 축하합니다! 갤럭시 S24 무료 증정 당첨 |
| 투자/주식 | 내일 상한가 확실한 종목 공개합니다 |
| 성인/불법 | 재택 고수익 알바 모집 — 하루 100만원 |
| 피싱/사칭 | [카카오] 계정 보안 이상 감지 — 즉시 확인 필요 |
| 도박 | 합법 카지노 가입 시 50만원 보너스 지급 |
| 광고/마케팅 | [광고] 홍삼 세트 최대 40% 할인 — 오늘만 |

**정상 (7종)**

| 카테고리 | 예시 제목 |
|---------|---------|
| 업무 | Q1 전략 기획 관련 검토 요청드립니다 |
| 공지/안내 | [공지] 3월 정기 시스템 점검 안내 |
| 뉴스레터 | 이번 주 AI/머신러닝 트렌드 정리 |
| 청구서/영수증 | [쿠팡] 3월 이용요금 청구 안내 |
| 배송/쇼핑 | [CJ대한통운] 주문하신 상품이 발송되었습니다 |
| 개인/지인 | 안녕하세요, 오랜만이에요 |
| 학교/교육 | [서울대학교] 3월 학사 일정 안내 |

---

## 평가 지표 (목표)

스팸 필터에서는 **정상 메일을 스팸으로 잘못 분류(False Positive)** 하는 것이 가장 큰 문제이므로 Precision을 우선적으로 관리합니다.

| 지표 | 목표 |
|------|------|
| Precision (스팸) | ≥ 0.98 |
| Recall (스팸) | ≥ 0.95 |
| F1-Score | ≥ 0.96 |

---

## 라이선스

MIT License
