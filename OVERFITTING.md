# 오버피팅 문제 해결 기록

## 문제 진단

초기 모델은 `korean_spam_dataset.csv` 단일 파일(5,000개)만으로 학습했습니다.  
이 데이터는 **직접 만든 합성 데이터**로, 스팸/햄 각각의 패턴이 너무 일정하게 반복되었습니다.  
그 결과 모델이 학습 데이터의 특정 패턴을 암기하게 되어 오버피팅이 발생했습니다.

### 오버피팅 징후

- Train Accuracy/F1 >> Test Accuracy/F1 (큰 갭)
- 새로운 스팸 메일에 대한 분류 성능 저하
- K-Fold 교차 검증 시 Fold 간 편차가 큼

---

## 해결 방법 5가지

### 1. 데이터 다양성 확보 (가장 큰 효과)

**문제:** 단일 출처 데이터는 특정 문체·어휘 패턴에 치우침  
**해결:** 독립적인 출처의 데이터를 추가해 총 8,000개로 확충

| 파일 | 수량 | 생성 방식 |
|------|------|----------|
| `korean_spam_dataset.csv` | 5,000개 | 수동 합성 (기존) |
| `claude_generated_ko.csv` | 3,000개 | 템플릿 기반 로컬 생성 (신규) |

**신규 데이터 구성:**
- 스팸 15종 × 100개: 피싱, 불법대출, 당첨사기, 투자사기, 취업사기, 보이스피싱, 성인광고, 의약품사기, 쇼핑광고, 해킹협박, 구독함정, 도박광고, 부동산스팸, 브랜드사칭, 해외스팸
- 햄 15종 × 100개: 업무연락, 회의일정, 프로젝트, 쇼핑영수증, 배송알림, 은행거래, 교육학교, 의료예약, 서비스가입, 뉴스레터, 가족친구, 소셜알림, 이벤트초대, 고객센터, 공공기관

**수정 파일:**
- `src/data/loader.py` — `load_datasets(csv_paths=[...])` 함수 추가
- `notebooks/01_data_exploration.ipynb` — 두 소스 통합 탐색 및 분포 시각화
- `notebooks/02_preprocessing.ipynb` — `load_datasets` 적용
- `notebooks/03_ml_model.ipynb` — DATA_PATHS 두 CSV 지정
- `notebooks/04_bert_model.ipynb` — DATA_PATHS 두 CSV 지정

---

### 2. 전처리 기반 특이값 제거

**문제:** 모델이 특정 URL, 전화번호, 이메일 주소를 스팸 지표로 암기  
**해결:** `Preprocessor`가 이 값들을 일반화된 토큰으로 치환

| 원본 | 치환 결과 | 효과 |
|------|-----------|------|
| `http://spam-site.com/abc?ref=xyz` | `[URL]` | 특정 URL 암기 방지 |
| `win@fake-prize.co.kr` | `[EMAIL]` | 특정 도메인 암기 방지 |
| `010-1234-5678` | `[PHONE]` | 특정 번호 패턴 암기 방지 |
| `$100,000` | `[PRICE]` | 특정 금액 암기 방지 |

토큰 치환 없이 학습하면 모델이 새로운 URL/번호를 가진 스팸은 탐지하지 못합니다.

**수정 파일:**
- `src/data/preprocessor.py` — 기존 구현 유지 (이미 적용됨)
- `notebooks/02_preprocessing.ipynb` — 토큰 치환이 오버피팅에 미치는 영향 설명 추가

---

### 3. K-Fold 교차 검증 (ML 모델)

**문제:** 단순 train/val/test 분리는 분할 방식에 따라 결과가 달라질 수 있음  
**해결:** `StratifiedKFold(n_splits=5)`로 전체 데이터를 균등하게 나눠 검증

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, all_X, all_y, cv=cv, scoring='f1', n_jobs=-1)
print(f"평균 F1: {scores.mean():.4f} ± {scores.std():.4f}")
```

- `std > 0.03` → Fold 간 편차 큼, 데이터 다양성 부족 신호
- `std ≤ 0.03` → 안정적인 일반화 성능

**수정 파일:**
- `notebooks/03_ml_model.ipynb` — K-Fold 교차 검증 섹션 추가

---

### 4. 학습 곡선 시각화 (ML 모델)

**문제:** 오버피팅 여부와 데이터를 더 늘려야 할지 판단이 어려움  
**해결:** `sklearn.model_selection.learning_curve`로 학습 데이터 양에 따른 성능 추이 시각화

```python
train_sizes, train_scores, val_scores = learning_curve(
    pipeline, all_X, all_y,
    cv=5, scoring='f1',
    train_sizes=np.linspace(0.1, 1.0, 8),
)
```

**그래프 해석:**

| 패턴 | 의미 | 조치 |
|------|------|------|
| Train ≈ Val, 둘 다 높음 | 정상 | 없음 |
| Train >> Val (큰 갭 유지) | 오버피팅 | C 값 낮추기, max_features 줄이기 |
| 두 선 모두 수렴하며 올라감 | 데이터 부족 | 데이터 더 추가 |

**수정 파일:**
- `notebooks/03_ml_model.ipynb` — 학습 곡선 섹션 추가

---

### 5. BERT Early Stopping (patience 기반)

**문제:** 기존에는 best checkpoint 복원만 있었고, 불필요한 에폭까지 학습을 계속함  
**해결:** `early_stopping_patience=3` 추가 — val F1이 3에폭 연속 개선되지 않으면 학습 중단

```python
bert = BERTSpamClassifier(
    epochs=10,
    early_stopping_patience=3,   # 추가된 파라미터
    weight_decay=0.01,           # L2 정규화
)
```

**동작 방식:**
1. 매 에폭 후 val F1 계산
2. 이전 최고 val F1보다 높으면 → best checkpoint 저장, `no_improve = 0` 초기화
3. 높지 않으면 → `no_improve += 1`
4. `no_improve >= patience` → 학습 조기 종료 후 best checkpoint 복원

**효과:** 과도한 학습으로 인한 오버피팅 방지 + 불필요한 학습 시간 절감

**수정 파일:**
- `src/models/bert_model.py` — `early_stopping_patience` 파라미터 및 로직 추가
- `notebooks/04_bert_model.ipynb` — 모델 생성 시 `early_stopping_patience=3` 명시

---

## 수정 파일 요약

| 파일 | 수정 내용 |
|------|----------|
| `src/data/loader.py` | `load_datasets()` 다중 CSV 로드 함수 추가 |
| `src/models/bert_model.py` | `early_stopping_patience` 파라미터 및 patience 기반 조기 종료 구현, `self.history` 추가 |
| `notebooks/01_data_exploration.ipynb` | 두 데이터셋 통합 로드, 소스별 분포 시각화 추가 |
| `notebooks/02_preprocessing.ipynb` | `load_datasets` 적용, 전처리-오버피팅 관계 설명 추가 |
| `notebooks/03_ml_model.ipynb` | 다중 CSV 로드, 오버피팅 진단(Train vs Test 비교), K-Fold CV, 학습 곡선 추가 |
| `notebooks/04_bert_model.ipynb` | 다중 CSV 로드, `early_stopping_patience=3` 적용, 학습 곡선 시각화 추가 |
| `notebooks/05_api_demo.ipynb` | 오버피팅 방지 적용 사항 요약 노트 추가 |
| `data/claude_generated_ko.csv` | 신규 생성 — 스팸 15종 + 햄 15종, 총 3,000개 |
| `_gen_ko_data.py` | 위 파일 생성 스크립트 |

---

## 적용 전후 비교

| 항목 | 적용 전 | 적용 후 |
|------|---------|---------|
| 학습 데이터 수 | 5,000개 (단일 출처) | 8,000개 (2개 출처) |
| 데이터 카테고리 수 | 스팸 7종 + 햄 7종 | 스팸 15종 + 햄 15종 이상 |
| 오버피팅 진단 방법 | 없음 | Train vs Test 비교, K-Fold, 학습 곡선 |
| BERT 조기 종료 | 없음 (전체 에폭 학습) | patience=3 early stopping |
| 교차 검증 | 없음 | 5-Fold Stratified CV |
