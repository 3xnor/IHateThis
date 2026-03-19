# 스팸 메일 분류 AI — 시스템 설계서

---

## 진행 현황

```
[1단계] 데이터셋 생성          ← 완료
[2단계] 전처리 파이프라인       ← 완료
[3단계] ML 베이스라인 모델      ← 완료
[4단계] BERT 파인튜닝           ← 완료 (학습 실행 필요)
[5단계] API 서버 & 배포         ← 완료 (모델 학습 후 실행)
```

---

## 1. 전체 아키텍처

```
[이메일 입력]
     │
     ▼
┌─────────────────┐
│  전처리 모듈     │  HTML 제거, 특수문자 정리,
│  Preprocessor   │  소문자 변환, 불용어 제거
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌──────────────┐
│ 경량   │ │ 고성능        │
│ ML     │ │ BERT 모델     │
│ 모델   │ │ (KR-ELECTRA) │
└───┬────┘ └──────┬───────┘
    │              │
    └──────┬───────┘
           ▼
   ┌───────────────┐
   │  분류 결과     │  spam / ham + confidence
   └───────┬───────┘
           ▼
   ┌───────────────┐
   │  FastAPI 서버  │  REST API로 외부 제공
   └───────────────┘
```

---

## 2. 데이터셋 (완료)

### 파일
- `generate_dataset.py` — 생성 스크립트
- `data/korean_spam_dataset.csv` — 생성 결과물

### 스펙

| 항목 | 내용 |
|------|------|
| 전체 | 5,000개 |
| 스팸 | 2,000개 (40%) |
| 정상 | 3,000개 (60%) |
| 컬럼 | id, subject, body, label, category |
| 인코딩 | UTF-8 (BOM) |

### 생성 방식
- 템플릿 기반 합성 데이터 (스팸 7종 / 정상 7종)
- 이름, 금액, 회사명, 날짜 등 무작위 치환으로 다양성 확보
- `random.seed(42)` — 재현 가능

### 카테고리별 분포

**스팸:** 대출/금융, 당첨/이벤트, 투자/주식, 성인/불법, 피싱/사칭, 도박, 광고/마케팅

**정상:** 업무, 공지/안내, 뉴스레터, 청구서/영수증, 배송/쇼핑, 개인/지인, 학교/교육

### 클래스 불균형
spam : ham = 4 : 6 으로 실제 이메일 환경을 반영.
학습 시 `class_weight='balanced'` 적용 예정.

---

## 3. 전처리 모듈 (`src/data/preprocessor.py`) — 예정

**입력:** 원시 이메일 텍스트 (제목 + 본문)
**출력:** 정제된 문자열

처리 순서:
1. HTML 태그 제거 (`BeautifulSoup`)
2. URL 패턴 → `[URL]` 토큰으로 치환
3. 이메일 주소 → `[EMAIL]` 토큰으로 치환
4. 전화번호 → `[PHONE]` 토큰으로 치환
5. 특수문자 정리 (과도한 기호 제거)
6. 소문자 변환 (영문)
7. 불용어 제거 (한국어: KoNLPy)
8. 공백 정규화

```python
class Preprocessor:
    def clean(self, text: str) -> str: ...
    def tokenize(self, text: str) -> list[str]: ...
    def preprocess(self, subject: str, body: str) -> str: ...
```

---

## 4. 경량 ML 모델 (`src/models/ml_model.py`) — 예정

빠른 추론이 필요하거나 GPU 없이 사용할 때 적합합니다.

**파이프라인:**
```
텍스트 → TF-IDF 벡터화 → Logistic Regression → 클래스 확률
```

**설정값 (config.yaml):**
```yaml
ml_model:
  tfidf:
    max_features: 50000
    ngram_range: [1, 2]   # 단어 + 바이그램
    sublinear_tf: true
  logistic_regression:
    C: 1.0
    max_iter: 1000
    class_weight: balanced
```

**대안 모델 (성능 비교용):**
- Naive Bayes, Random Forest, SVM (LinearSVC)

---

## 5. BERT 기반 모델 (`src/models/bert_model.py`) — 예정

한국어 데이터셋이므로 `snunlp/KR-ELECTRA-discriminator` 사용 예정.

**구조:**
```
[CLS] 제목 [SEP] 본문 [SEP]
        │
   BERT Encoder
        │
   [CLS] 벡터 (768차원)
        │
   Dropout(0.3)
        │
   Linear(768 → 2)
        │
   Softmax → [ham 확률, spam 확률]
```

**학습 설정:**
```yaml
bert_model:
  base_model: "snunlp/KR-ELECTRA-discriminator"
  max_length: 256
  batch_size: 32
  epochs: 5
  learning_rate: 2e-5
  warmup_ratio: 0.1
  weight_decay: 0.01
```

---

## 6. 학습 파이프라인 (`src/train.py`) — 예정

```
data/korean_spam_dataset.csv 로드
    │
    ├─ 전처리 (Preprocessor)
    │
    ├─ train / validation / test 분리 (8 : 1 : 1)
    │
    ├─ 모델 학습
    │    └─ BERT: 에폭마다 validation F1 체크, Early Stopping
    │
    ├─ 평가 (test set)
    │
    └─ 모델 저장 (artifacts/)
```

---

## 7. API 서버 (`src/api/`) — 예정

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/predict` | 단일 이메일 분류 |
| POST | `/predict/batch` | 다수 이메일 일괄 분류 |
| GET | `/health` | 서버 상태 확인 |
| GET | `/model/info` | 현재 모델 정보 |

**요청:**
```python
class PredictRequest(BaseModel):
    subject: str
    body: str
    model: Literal["ml", "bert"] = "bert"
```

**응답:**
```python
class PredictResponse(BaseModel):
    label: Literal["spam", "ham"]
    confidence: float
    spam_probability: float
    ham_probability: float
    model: str
    processing_time_ms: float
```

---

## 8. 성능 목표

| 지표 | 목표 |
|------|------|
| Precision (스팸) | ≥ 0.98 |
| Recall (스팸) | ≥ 0.95 |
| F1-Score | ≥ 0.96 |

스팸 판정 임계값은 FP(정상 → 스팸 오분류) 억제를 위해 0.7로 설정.

```python
SPAM_THRESHOLD = 0.7
```

### 추론 속도 목표

| 모델 | 목표 처리 시간 |
|------|--------------|
| ML (TF-IDF + LR) | ~5ms |
| BERT (CPU) | ~200ms |
| BERT (GPU) | ~20ms |
