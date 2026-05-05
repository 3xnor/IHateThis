"""
BERT 기반 모델 — KR-ELECTRA 파인튜닝
높은 정확도가 필요할 때 사용하는 고성능 모델입니다.
GPU 없이도 동작하지만 추론 속도가 느립니다.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

from src.models.base import BaseSpamClassifier


class EmailDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int] | None,
        tokenizer,
        max_length: int,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class BERTSpamClassifier(BaseSpamClassifier):
    """
    KR-ELECTRA 기반 스팸 분류기.
    AutoModelForSequenceClassification을 사용하여
    [CLS] 벡터 → Linear(768→2) → Softmax 구조입니다.
    """

    def __init__(
        self,
        base_model: str = "snunlp/KR-ELECTRA-discriminator",
        max_length: int = 256,
        batch_size: int = 32,
        epochs: int = 5,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        spam_threshold: float = 0.7,
        early_stopping_patience: int = 3,
    ) -> None:
        self.base_model = base_model
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.SPAM_THRESHOLD = spam_threshold
        self.early_stopping_patience = early_stopping_patience
        self.history: dict[str, list] = {"train_loss": [], "val_f1": []}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[BERT] 디바이스: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            base_model, num_labels=2
        ).to(self.device)

    # ------------------------------------------------------------------
    # BaseSpamClassifier 구현
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: list[str],
        y_train: list[int],
        X_val: list[str] | None = None,
        y_val: list[int] | None = None,
    ) -> None:
        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader = self._make_loader(X_val, y_val) if X_val else None

        total_steps = len(train_loader) * self.epochs
        warmup_steps = int(total_steps * self.warmup_ratio)

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        best_val_f1 = 0.0
        best_state = None
        no_improve = 0
        self.history = {"train_loss": [], "val_f1": []}

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(train_loader, optimizer, scheduler)
            self.history["train_loss"].append(train_loss)
            print(f"[BERT] Epoch {epoch}/{self.epochs} | train_loss={train_loss:.4f}", end="")

            if val_loader:
                val_f1 = self._evaluate_epoch(val_loader)
                self.history["val_f1"].append(val_f1)
                print(f" | val_f1={val_f1:.4f}", end="")
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.early_stopping_patience:
                        print(f"\n[BERT] Early stopping (patience={self.early_stopping_patience}, epoch={epoch})")
                        break
            print()

        # 최적 가중치 복원
        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"[BERT] 최적 모델 복원 (val_f1={best_val_f1:.4f})")

    def predict_proba(self, texts: list[str]) -> list[tuple[float, float]]:
        loader = self._make_loader(texts, labels=None, shuffle=False)
        self.model.eval()
        results = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                probs = torch.softmax(logits, dim=-1).cpu().tolist()
                results.extend([(p[0], p[1]) for p in probs])
        return results

    def save(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"[BERT] 모델 저장: {path}")

    @classmethod
    def load(cls, path: str) -> "BERTSpamClassifier":
        obj = cls.__new__(cls)
        obj.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        obj.base_model = path
        obj.SPAM_THRESHOLD = 0.7
        obj.max_length = 256
        obj.batch_size = 32
        obj.tokenizer = AutoTokenizer.from_pretrained(path)
        obj.model = AutoModelForSequenceClassification.from_pretrained(path).to(obj.device)
        print(f"[BERT] 모델 로드: {path}")
        return obj

    # ------------------------------------------------------------------
    # 내부 메서드
    # ------------------------------------------------------------------

    def _make_loader(
        self,
        texts: list[str],
        labels: list[int] | None,
        shuffle: bool = False,
    ) -> DataLoader:
        dataset = EmailDataset(texts, labels, self.tokenizer, self.max_length)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def _train_epoch(self, loader: DataLoader, optimizer, scheduler) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            loss = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            ).loss
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def _evaluate_epoch(self, loader: DataLoader) -> float:
        from sklearn.metrics import f1_score

        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                logits = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits
                preds = torch.argmax(logits, dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(batch["labels"].tolist())
        return f1_score(all_labels, all_preds, pos_label=1)

    @property
    def model_name(self) -> str:
        return f"KR-ELECTRA ({self.base_model})"
