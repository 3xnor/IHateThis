# 클래스 다이어그램 — 한국어 스팸 메일 분류 AI

## 사용 방법
> **https://mermaid.live** 접속 → 왼쪽 에디터 전체 선택 후 아래 코드 붙여넣기

---

## 클래스 도출 근거

| 클래스 | 도출 근거 |
|--------|-----------|
| `BaseSpamClassifier` | 공통 분류 인터페이스 추상화 — ML/BERT 모델의 일반화 |
| `MLSpamClassifier` | TF-IDF + Logistic Regression 경량 분류기 |
| `BERTSpamClassifier` | KR-ELECTRA 기반 고성능 분류기 |
| `EmailDataset` | BERT 학습용 PyTorch 데이터셋 래퍼 |
| `Preprocessor` | 이메일 텍스트 정제 담당 단일 책임 클래스 |
| `DataSplit` | 데이터 분리 결과 보관 값 객체 (Value Object) |
| `PredictResult` | 예측 결과 보관 값 객체 |
| `SpamClassifierAPI` | REST API 진입점 — 모델·전처리기 조합 컨트롤러 |
| `PredictRequest` | 단건 분류 API 요청 스키마 |
| `PredictResponse` | 단건 분류 API 응답 스키마 |
| `BatchPredictRequest` | 다건 분류 API 요청 스키마 |
| `BatchPredictResponse` | 다건 분류 API 응답 스키마 |
| `HealthResponse` | 서버 상태 확인 응답 스키마 |
| `ModelInfoResponse` | 모델 정보 조회 응답 스키마 |

---

## 클래스 다이어그램

```
classDiagram
    direction TB

    %% ════════════════════════════════════
    %% 모델 계층
    %% ════════════════════════════════════

    class BaseSpamClassifier {
        <<abstract>>
        +float SPAM_THRESHOLD
        +fit(X_train: list, y_train: list, X_val: list, y_val: list) void
        +predict_proba(texts: list) list
        +predict(texts: list) list
        +predict_single(text: str) PredictResult
        +save(path: str) void
        +load(path: str) BaseSpamClassifier
    }

    class MLSpamClassifier {
        -Pipeline _pipeline
        +float SPAM_THRESHOLD
        +fit(X_train: list, y_train: list, X_val: list, y_val: list) void
        +predict_proba(texts: list) list
        +save(path: str) void
        +load(path: str) MLSpamClassifier
        +model_name() str
    }

    class BERTSpamClassifier {
        +str base_model
        +int max_length
        +int batch_size
        +int epochs
        +float learning_rate
        +float warmup_ratio
        +float weight_decay
        +float SPAM_THRESHOLD
        -device device
        -tokenizer tokenizer
        -model model
        +fit(X_train: list, y_train: list, X_val: list, y_val: list) void
        +predict_proba(texts: list) list
        +save(path: str) void
        +load(path: str) BERTSpamClassifier
        +model_name() str
        -_make_loader(texts: list, labels: list, shuffle: bool) DataLoader
        -_train_epoch(loader: DataLoader, optimizer, scheduler) float
        -_evaluate_epoch(loader: DataLoader) float
    }

    class EmailDataset {
        -list texts
        -list labels
        -tokenizer tokenizer
        -int max_length
        +__len__() int
        +__getitem__(idx: int) dict
    }

    %% ════════════════════════════════════
    %% 데이터 계층
    %% ════════════════════════════════════

    class Preprocessor {
        +bool use_konlpy
        -analyzer _analyzer
        +clean(text: str) str
        +tokenize(text: str) list
        +preprocess(subject: str, body: str) str
        -_remove_html(text: str) str
        -_replace_urls(text: str) str
        -_replace_emails(text: str) str
        -_replace_phones(text: str) str
        -_clean_special(text: str) str
        -_normalize_whitespace(text: str) str
    }

    class DataSplit {
        +list X_train
        +list X_val
        +list X_test
        +list y_train
        +list y_val
        +list y_test
    }

    class PredictResult {
        +str label
        +float confidence
        +float spam_probability
        +float ham_probability
    }

    %% ════════════════════════════════════
    %% API 계층
    %% ════════════════════════════════════

    class SpamClassifierAPI {
        -dict _models
        -Preprocessor _preprocessor
        -dict _cfg
        +health() HealthResponse
        +model_info() ModelInfoResponse
        +predict(req: PredictRequest) PredictResponse
        +predict_batch(req: BatchPredictRequest) BatchPredictResponse
    }

    class PredictRequest {
        +str subject
        +str body
        +str model
    }

    class PredictResponse {
        +str label
        +float confidence
        +float spam_probability
        +float ham_probability
        +str model
        +float processing_time_ms
    }

    class BatchPredictRequest {
        +list emails
    }

    class BatchPredictResponse {
        +list results
        +int total
        +float total_processing_time_ms
    }

    class HealthResponse {
        +str status
        +bool ml_model_loaded
        +bool bert_model_loaded
    }

    class ModelInfoResponse {
        +str ml_model
        +str bert_model
        +float spam_threshold
    }

    %% ════════════════════════════════════
    %% 관계 정의
    %% ════════════════════════════════════

    %% 일반화 (상속)
    BaseSpamClassifier <|-- MLSpamClassifier : 일반화
    BaseSpamClassifier <|-- BERTSpamClassifier : 일반화

    %% 의존 관계
    BaseSpamClassifier ..> PredictResult : «반환»

    %% 컴포지션 (강한 포함)
    BERTSpamClassifier "1" *-- "0..*" EmailDataset : 생성
    BatchPredictRequest "1" *-- "1..100" PredictRequest : 포함
    BatchPredictResponse "1" *-- "1..*" PredictResponse : 포함

    %% 연관 관계
    SpamClassifierAPI "1" --> "0..1" MLSpamClassifier : ml_model 보유
    SpamClassifierAPI "1" --> "0..1" BERTSpamClassifier : bert_model 보유
    SpamClassifierAPI "1" --> "1" Preprocessor : 전처리 위임

    %% 의존 관계 (API ↔ 스키마)
    SpamClassifierAPI ..> PredictRequest : «요청 수신»
    SpamClassifierAPI ..> PredictResponse : «응답 반환»
    SpamClassifierAPI ..> BatchPredictRequest : «배치 요청 수신»
    SpamClassifierAPI ..> BatchPredictResponse : «배치 응답 반환»
    SpamClassifierAPI ..> HealthResponse : «상태 응답»
    SpamClassifierAPI ..> ModelInfoResponse : «모델 정보 응답»
```

---

## 관계 요약표

| 관계 종류 | 출발 클래스 | 도착 클래스 | 연관명 / 의존명 | 다중성 |
|-----------|-------------|-------------|----------------|--------|
| 일반화 (상속) | `MLSpamClassifier` | `BaseSpamClassifier` | 일반화 | — |
| 일반화 (상속) | `BERTSpamClassifier` | `BaseSpamClassifier` | 일반화 | — |
| 의존 | `BaseSpamClassifier` | `PredictResult` | «반환» | 1 → 1..* |
| 컴포지션 | `BERTSpamClassifier` | `EmailDataset` | 생성 | 1 → 0..* |
| 컴포지션 | `BatchPredictRequest` | `PredictRequest` | 포함 | 1 → 1..100 |
| 컴포지션 | `BatchPredictResponse` | `PredictResponse` | 포함 | 1 → 1..* |
| 연관 | `SpamClassifierAPI` | `MLSpamClassifier` | ml_model 보유 | 1 → 0..1 |
| 연관 | `SpamClassifierAPI` | `BERTSpamClassifier` | bert_model 보유 | 1 → 0..1 |
| 연관 | `SpamClassifierAPI` | `Preprocessor` | 전처리 위임 | 1 → 1 |
| 의존 | `SpamClassifierAPI` | `PredictRequest` | «요청 수신» | 1 → 1 |
| 의존 | `SpamClassifierAPI` | `PredictResponse` | «응답 반환» | 1 → 1 |
| 의존 | `SpamClassifierAPI` | `BatchPredictRequest` | «배치 요청 수신» | 1 → 1 |
| 의존 | `SpamClassifierAPI` | `BatchPredictResponse` | «배치 응답 반환» | 1 → 1 |
| 의존 | `SpamClassifierAPI` | `HealthResponse` | «상태 응답» | 1 → 1 |
| 의존 | `SpamClassifierAPI` | `ModelInfoResponse` | «모델 정보 응답» | 1 → 1 |
