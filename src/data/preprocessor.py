"""
전처리 모듈
이메일 텍스트를 정제하여 모델 입력 형태로 변환합니다.
"""

import re
import string
from bs4 import BeautifulSoup


class Preprocessor:
    """
    이메일 전처리 클래스.

    처리 순서:
    1. HTML 태그 제거
    2. URL → [URL] 토큰 치환
    3. 이메일 주소 → [EMAIL] 토큰 치환
    4. 전화번호 → [PHONE] 토큰 치환
    5. 특수문자 정리
    6. 영문 소문자 변환
    7. 불용어 제거 (KoNLPy 사용 시)
    8. 공백 정규화
    """

    # 패턴 상수
    _URL_PATTERN = re.compile(
        r"https?://\S+|www\.\S+|bit\.ly/\S+|goo\.gl/\S+",
        re.IGNORECASE,
    )
    _EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
    _PHONE_PATTERN = re.compile(
        r"(?:\+?82[-\s]?)?0\d{1,2}[-\s]?\d{3,4}[-\s]?\d{4}"
    )
    _EXCESS_SPECIAL = re.compile(r"[^\w\s가-힣ㄱ-ㅎㅏ-ㅣ\[\]]")
    _WHITESPACE = re.compile(r"\s+")

    # 한국어 불용어 (KoNLPy 미사용 시 기본 목록)
    _KOREAN_STOPWORDS = {
        "이", "그", "저", "것", "수", "등", "및", "의", "가", "을", "를",
        "에", "는", "은", "이", "도", "로", "으로", "와", "과", "하다",
        "있다", "되다", "않다", "없다", "같다", "위해", "통해", "대한",
        "관련", "경우", "때", "또한", "그리고", "하지만", "그러나",
    }

    def __init__(self, use_konlpy: bool = False, konlpy_analyzer: str = "Okt"):
        self.use_konlpy = use_konlpy
        self._analyzer = None

        if use_konlpy:
            self._init_konlpy(konlpy_analyzer)

    def _init_konlpy(self, analyzer_name: str) -> None:
        try:
            from konlpy.tag import Okt, Kkma, Komoran  # noqa: F401

            analyzers = {"Okt": Okt, "Kkma": Kkma, "Komoran": Komoran}
            cls = analyzers.get(analyzer_name, Okt)
            self._analyzer = cls()
        except ImportError:
            print("[경고] KoNLPy가 설치되어 있지 않습니다. 기본 불용어 목록을 사용합니다.")
            self.use_konlpy = False

    # ------------------------------------------------------------------
    # 공개 API
    # ------------------------------------------------------------------

    def clean(self, text: str) -> str:
        """HTML 제거 → 토큰 치환 → 특수문자 정리 → 소문자 → 공백 정규화"""
        if not isinstance(text, str):
            return ""

        text = self._remove_html(text)
        text = self._replace_urls(text)
        text = self._replace_emails(text)
        text = self._replace_phones(text)
        text = self._clean_special(text)
        text = text.lower()
        text = self._normalize_whitespace(text)
        return text

    def tokenize(self, text: str) -> list[str]:
        """정제된 텍스트를 토큰 리스트로 변환합니다."""
        if self.use_konlpy and self._analyzer is not None:
            tokens = self._analyzer.morphs(text)
            return [t for t in tokens if t not in self._KOREAN_STOPWORDS and len(t) > 1]
        else:
            tokens = text.split()
            return [t for t in tokens if t not in self._KOREAN_STOPWORDS and len(t) > 1]

    def preprocess(self, subject: str, body: str) -> str:
        """제목과 본문을 합쳐 전처리된 문자열을 반환합니다."""
        combined = f"{subject} {body}"
        cleaned = self.clean(combined)
        return cleaned

    # ------------------------------------------------------------------
    # 내부 메서드
    # ------------------------------------------------------------------

    @staticmethod
    def _remove_html(text: str) -> str:
        if "<" in text and ">" in text:
            soup = BeautifulSoup(text, "html.parser")
            return soup.get_text(separator=" ")
        return text

    @classmethod
    def _replace_urls(cls, text: str) -> str:
        return cls._URL_PATTERN.sub("[URL]", text)

    @classmethod
    def _replace_emails(cls, text: str) -> str:
        return cls._EMAIL_PATTERN.sub("[EMAIL]", text)

    @classmethod
    def _replace_phones(cls, text: str) -> str:
        return cls._PHONE_PATTERN.sub("[PHONE]", text)

    @classmethod
    def _clean_special(cls, text: str) -> str:
        # 한글, 영문, 숫자, 공백, 대괄호(토큰용)만 남김
        text = re.sub(r"[^\w\s가-힣ㄱ-ㅎㅏ-ㅣ\[\]]", " ", text)
        return text

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()
