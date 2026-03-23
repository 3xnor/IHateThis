"""
한국어 스팸/정상 이메일 데이터셋 생성기
실행: python generate_dataset.py
출력: data/korean_spam_dataset.csv (5000개)
"""

import csv
import random
import os
from datetime import datetime, timedelta

random.seed(42)

# ─────────────────────────────────────────────
# 스팸 템플릿
# ─────────────────────────────────────────────

SPAM_TEMPLATES = {
    "대출/금융": {
        "subjects": [
            "{name}님 {amount}만원 즉시 대출 가능합니다",
            "신용등급 상관없이 {amount}만원 대출 승인!",
            "오늘만 특별 저금리 {rate}% 대출 안내",
            "[긴급] {name}님 대출 한도 조회 결과",
            "무직자/주부도 가능! {amount}만원 당일 대출",
            "금리 {rate}% 최저금리 대환대출 안내",
            "신용불량자도 OK! 비대면 {amount}만원 대출",
        ],
        "bodies": [
            "{name}님 안녕하세요.\n\n저희 {company} 대출 서비스를 이용해 보세요.\n최대 {amount}만원까지 당일 지급 가능합니다.\n\n✔ 금리: 연 {rate}%\n✔ 한도: 최대 {amount}만원\n✔ 심사시간: 10분 이내\n\n지금 바로 신청하세요! ▶ {fake_url}\n\n수신거부: 080-{phone}",
            "안녕하세요, {company}입니다.\n\n{name}님의 신용점수를 조회한 결과,\n{amount}만원 대출이 가능하십니다.\n\n지금 신청하시면 {rate}% 특별금리 적용!\n\n☎ 상담전화: 0{phone}\n▶ 신청링크: {fake_url}",
            "【{company}】 {name}님 전용 대출 상품 안내\n\n현재 보유하신 신용점수로 {amount}만원 즉시 대출 가능합니다.\n\n- 최저금리 {rate}%\n- 중도상환수수료 없음\n- 24시간 신청 가능\n\n▶ 클릭: {fake_url}",
        ],
    },
    "당첨/이벤트": {
        "subjects": [
            "[당첨] {name}님이 {prize} 이벤트에 당첨되었습니다!",
            "축하합니다! {prize} 무료 증정 이벤트 당첨",
            "{name}님 {prize} 당첨 확인 바랍니다",
            "오늘만! 선착순 {count}명 {prize} 100% 증정",
            "당신이 선택되었습니다 — {prize} 수령 안내",
            "이벤트 참여 시 {prize} 100% 지급",
        ],
        "bodies": [
            "안녕하세요, {name}님!\n\n축하드립니다! 저희 {company} 이벤트에 당첨되셨습니다.\n\n당첨 상품: {prize}\n수령 기한: {deadline}까지\n\n지금 바로 수령하세요 ▶ {fake_url}\n\n※ 기한 내 미수령 시 당첨이 취소됩니다.",
            "{name}님, 특별히 선발되셨습니다!\n\n{company}에서 {prize}를 무료로 드립니다.\n단, {deadline}까지 아래 링크에서 주소를 등록하셔야 합니다.\n\n▶ {fake_url}\n\n개인정보 수집 동의 후 수령 가능합니다.",
            "【무료증정】{prize} 당첨 안내\n\n안녕하세요 {name}님,\n이번 달 {company} 이벤트 당첨자로 선정되셨습니다!\n\n상품: {prize}\n발송일: {deadline}\n\n수령을 원하시면 클릭해 주세요: {fake_url}",
        ],
    },
    "투자/주식": {
        "subjects": [
            "내일 상한가 확실한 종목 공개합니다",
            "[무료] {stock} 종목 매수 타이밍 알림",
            "월 {amount}만원 수익 보장 투자 시스템",
            "전문가 픽! 이번 주 급등 예상 종목",
            "{name}님만을 위한 VIP 투자 정보",
            "코인 자동매매로 하루 {rate}% 수익 실현",
        ],
        "bodies": [
            "{name}님 안녕하세요.\n\n저희 {company} 리서치팀이 분석한 급등 예상 종목을 공개합니다.\n\n📈 종목명: {stock}\n📊 목표가: {amount}원\n⏰ 매수 타이밍: 내일 장 시작 직후\n\n자세한 정보: {fake_url}\n\n※ 투자는 본인 책임입니다.",
            "안녕하세요!\n\n매달 안정적으로 {amount}만원을 버는 방법을 알려드립니다.\n실제 수익 인증 가능! 거짓말 아닙니다.\n\n지금 무료 가입 후 {prize} 받아가세요.\n▶ {fake_url}\n\n수신거부 {fake_url}/unsubscribe",
            "【긴급 투자 정보】\n\n{stock} 종목이 곧 급등할 것으로 예상됩니다.\n\n- 현재가: {amount}원\n- 목표가: {amount2}원 (+{rate}%)\n\n지금 바로 확인하세요: {fake_url}",
        ],
    },
    "성인/불법": {
        "subjects": [
            "[성인] 오늘 밤 {name}님과 만날 수 있어요",
            "비밀 만남 주선 — 지금 가입하면 무료",
            "나이트 알바 구인 일당 {amount}만원",
            "재택 고수익 알바 모집 — 하루 {amount}만원",
            "[비밀] {name}님에게 메시지가 도착했습니다",
        ],
        "bodies": [
            "{name}님 안녕하세요.\n\n회원님 근처에 만남을 원하는 분이 계십니다.\n지금 가입하면 무료로 연락처를 드립니다.\n\n▶ {fake_url}\n\n성인인증 후 이용 가능합니다.",
            "안녕하세요!\n\n재택으로 하루 {amount}만원 버는 방법 알려드립니다.\n학력/경력 무관, 스마트폰만 있으면 OK!\n\n자세한 내용: {fake_url}",
        ],
    },
    "피싱/사칭": {
        "subjects": [
            "[카카오] 계정 보안 이상 감지 — 즉시 확인 필요",
            "[네이버] 비정상 로그인 시도 감지",
            "[국세청] {name}님 세금 환급금 {amount}원 안내",
            "[건강보험공단] 미납 보험료 납부 안내",
            "[법원] 민사소송 출석 요구서 발송",
            "[금감원] 명의도용 피해 확인 요청",
        ],
        "bodies": [
            "안녕하세요, {name}님.\n\n[카카오 고객센터]\n\n회원님의 계정에서 비정상적인 접속이 감지되었습니다.\n보안을 위해 즉시 아래 링크에서 본인인증을 해주세요.\n\n▶ 본인인증 링크: {fake_url}\n\n24시간 내 인증하지 않으면 계정이 잠금 처리됩니다.",
            "{name}님께\n\n국세청에서 알려드립니다.\n\n귀하의 세금 환급금 {amount}원이 미처리 상태입니다.\n아래 링크에서 계좌를 등록하시면 3영업일 내 입금됩니다.\n\n▶ 환급 신청: {fake_url}\n\n문의: 국세청 고객센터 126",
            "안녕하세요.\n\n[건강보험공단] 안내말씀 드립니다.\n\n2024년 보험료 {amount}원이 미납 상태입니다.\n납부하지 않으실 경우 불이익이 발생할 수 있습니다.\n\n▶ 즉시 납부: {fake_url}",
        ],
    },
    "도박": {
        "subjects": [
            "합법 카지노 가입 시 {amount}만원 보너스 지급",
            "스포츠 토토 {rate}% 적중률 보장",
            "온라인 바카라 첫 충전 {amount}% 보너스",
            "해외 합법 카지노 VIP 초대",
        ],
        "bodies": [
            "안녕하세요!\n\n저희 {company} 카지노에 가입하시면 {amount}만원 보너스를 즉시 드립니다.\n\n✔ 24시간 운영\n✔ 즉시 환전 가능\n✔ 신규 가입 {amount}만원 지급\n\n▶ 가입하기: {fake_url}",
            "{name}님 안녕하세요.\n\n저희 {company}의 스포츠 분석가가 오늘 경기를 분석했습니다.\n\n적중률 {rate}% 이상 보장!\n지금 가입하면 첫 베팅 무료.\n\n▶ {fake_url}",
        ],
    },
    "광고/마케팅": {
        "subjects": [
            "[광고] {product} 최대 {rate}% 할인 — 오늘만",
            "{product} 공동구매 마감 임박!",
            "지금 구매 시 {product} 1+1 증정",
            "[특가] {product} {amount}원에 드립니다",
        ],
        "bodies": [
            "{name}님 안녕하세요.\n\n저희 {company}에서 {product}를 특가로 제공합니다!\n\n✔ 정가: {amount2}원\n✔ 특가: {amount}원 ({rate}% 할인)\n✔ 기간: {deadline}까지\n\n▶ 구매하기: {fake_url}\n\n수신거부: {fake_url}/unsubscribe",
            "안녕하세요, {company}입니다.\n\n이번 주말까지 {product} 공동구매를 진행합니다.\n\n참여 방법:\n1. 아래 링크 클릭\n2. 수량 선택\n3. 결제 완료\n\n▶ {fake_url}\n\n재고 소진 시 조기 마감될 수 있습니다.",
        ],
    },
}

# ─────────────────────────────────────────────
# 정상(ham) 템플릿
# ─────────────────────────────────────────────

HAM_TEMPLATES = {
    "업무": {
        "subjects": [
            "{project} 관련 검토 요청드립니다",
            "내일 {time} 회의 일정 확인 부탁드립니다",
            "{project} 진행 현황 공유",
            "Re: {project} 수정 사항 반영 건",
            "{deadline} 마감 — {document} 제출 안내",
            "{project} 관련 자료 첨부합니다",
            "금주 업무 보고 드립니다",
        ],
        "bodies": [
            "안녕하세요, {name}님.\n\n{project} 관련하여 검토가 필요한 사항이 있어 연락드립니다.\n\n첨부 파일을 확인하시고 {deadline}까지 피드백 부탁드립니다.\n\n감사합니다.\n{sender} 드림",
            "{name} 님,\n\n안녕하세요.\n\n내일 {time}에 {project} 관련 회의가 예정되어 있습니다.\n참석 가능 여부를 알려주시면 감사하겠습니다.\n\n장소: {place}\n안건: {project} 진행 방향 논의\n\n감사합니다.\n{sender}",
            "안녕하세요.\n\n{project} 진행 현황을 공유드립니다.\n\n- 완료: {task1}\n- 진행 중: {task2}\n- 예정: {task3}\n\n문의 사항 있으시면 말씀해 주세요.\n\n{sender} 올림",
        ],
    },
    "공지/안내": {
        "subjects": [
            "[공지] {month}월 정기 시스템 점검 안내",
            "[안내] {service} 이용 약관 개정 안내",
            "[중요] 개인정보 처리방침 변경 안내",
            "[공지] 사무실 {event} 안내",
            "{service} 서비스 업데이트 안내 ({version})",
        ],
        "bodies": [
            "안녕하세요.\n\n{service} 운영팀입니다.\n\n{month}월 정기 점검이 아래와 같이 진행됩니다.\n\n- 일시: {deadline}\n- 점검 내용: 서버 안정화 및 보안 패치\n- 영향: 서비스 일시 중단\n\n이용에 불편을 드려 죄송합니다.\n\n{service} 운영팀 드림",
            "안녕하세요, {name}님.\n\n{service}의 이용약관이 {deadline}부로 개정됩니다.\n\n주요 변경 사항:\n1. 개인정보 보유 기간 변경\n2. 제3자 제공 항목 수정\n\n자세한 내용은 홈페이지에서 확인하실 수 있습니다.\n\n감사합니다.",
        ],
    },
    "뉴스레터": {
        "subjects": [
            "[{company} 뉴스레터] {month}월 {week}주차 소식",
            "{company} 주간 업데이트 — {topic}",
            "이번 주 {topic} 트렌드 정리",
            "[{company}] {topic} 관련 최신 아티클 모음",
        ],
        "bodies": [
            "안녕하세요, {name}님!\n\n이번 주 {topic} 분야에서 주목할 만한 소식을 정리했습니다.\n\n1. {news1}\n2. {news2}\n3. {news3}\n\n더 자세한 내용은 홈페이지에서 확인하세요.\n\n{company} 뉴스레터팀 드림\n\n수신 거부를 원하시면 설정에서 변경하실 수 있습니다.",
            "{company} 주간 뉴스레터입니다.\n\n▶ 이번 주 주요 내용\n\n- {news1}\n- {news2}\n- {news3}\n\n질문이나 의견이 있으시면 언제든지 연락해 주세요.\n\n감사합니다.",
        ],
    },
    "청구서/영수증": {
        "subjects": [
            "[{company}] {month}월 이용요금 청구 안내",
            "{service} 결제 완료 안내 ({amount}원)",
            "[영수증] {product} 구매 확인",
            "{month}월 {service} 구독 결제 완료",
        ],
        "bodies": [
            "안녕하세요, {name}님.\n\n{month}월 이용요금 청구 내역을 안내드립니다.\n\n- 서비스명: {service}\n- 청구금액: {amount}원\n- 납부기한: {deadline}\n- 납부방법: 자동이체\n\n문의사항은 고객센터(1588-{phone})로 연락 주세요.\n\n감사합니다, {company}",
            "{name}님, 결제가 완료되었습니다.\n\n주문번호: {order_id}\n상품명: {product}\n결제금액: {amount}원\n결제일시: {deadline}\n\n이용해 주셔서 감사합니다.\n{company} 드림",
        ],
    },
    "배송/쇼핑": {
        "subjects": [
            "[{company}] 주문하신 상품이 발송되었습니다",
            "배송 완료 안내 — {product}",
            "{product} 주문 확인 안내",
            "[배송추적] {product} 배송 현황",
        ],
        "bodies": [
            "안녕하세요, {name}님.\n\n주문하신 상품이 발송되었습니다.\n\n- 주문번호: {order_id}\n- 상품명: {product}\n- 택배사: CJ대한통운\n- 운송장번호: {tracking_id}\n\n배송 조회는 택배사 홈페이지에서 가능합니다.\n\n감사합니다.",
            "{name}님의 {product} 주문이 확인되었습니다.\n\n주문 정보:\n주문번호: {order_id}\n결제금액: {amount}원\n배송지: {address}\n예상 배송일: {deadline}\n\n문의사항은 1:1 채팅으로 연락 주세요.\n{company} 고객센터",
        ],
    },
    "개인/지인": {
        "subjects": [
            "안녕하세요, 오랜만이에요",
            "{event} 관련해서 연락드려요",
            "다음 주 약속 어떠세요?",
            "자료 공유드립니다",
            "Re: 지난번 말씀하신 건 관련해서요",
        ],
        "bodies": [
            "안녕하세요, {name}님!\n\n오랜만에 연락드립니다.\n요즘 어떻게 지내세요?\n\n다음 주에 시간이 되시면 한번 뵙고 싶은데, 언제가 편하신지 알려주세요.\n\n감사합니다.",
            "{name}님, 안녕하세요.\n\n지난번에 부탁하셨던 자료 첨부해서 보내드립니다.\n\n혹시 추가로 필요한 것이 있으면 말씀해 주세요.\n\n즐거운 하루 되세요!",
            "안녕하세요!\n\n{event}와 관련해서 몇 가지 여쭤보고 싶은 게 있어서요.\n\n시간 되실 때 답변 주시면 감사하겠습니다.\n\n좋은 하루 보내세요 :)",
        ],
    },
    "학교/교육": {
        "subjects": [
            "[{school}] {month}월 학사 일정 안내",
            "{subject} 과제 제출 안내 — {deadline}까지",
            "[{school}] 장학금 신청 안내",
            "수강신청 안내 — {semester}학기",
            "[공지] {event} 행사 안내",
        ],
        "bodies": [
            "안녕하세요, {name}님.\n\n{school} 학사지원팀입니다.\n\n{month}월 학사 일정을 안내드립니다.\n\n- {task1}: {deadline}\n- {task2}: {deadline}\n- {task3}: {deadline}\n\n자세한 사항은 학교 홈페이지를 참고해 주세요.\n\n{school} 드림",
            "{name}님,\n\n{subject} 과제 제출 기한이 {deadline}입니다.\n\n제출 방법: LMS 시스템 업로드\n파일 형식: PDF 또는 Word\n\n기한 내 미제출 시 0점 처리될 수 있으니 주의하시기 바랍니다.\n\n감사합니다.",
        ],
    },
}

# ─────────────────────────────────────────────
# 랜덤 데이터 풀
# ─────────────────────────────────────────────

NAMES = ["김민준", "이서연", "박지호", "최수아", "정우진", "강하은", "윤도현", "임나연",
         "한서준", "오지은", "신민서", "배준혁", "류채원", "송현우", "문지아", "권태양"]
COMPANIES_SPAM = ["금융나라", "빠른대출", "머니플러스", "OK캐피탈", "스마트론", "행운복권",
                  "베스트주식", "코인킹", "럭키카지노", "톱광고", "이벤트나라"]
COMPANIES_HAM = ["삼성전자", "카카오", "네이버", "LG전자", "현대자동차", "SK텔레콤",
                 "CJ대한통운", "롯데쇼핑", "신한은행", "KB국민은행", "쿠팡", "배달의민족"]
SENDERS = ["김팀장", "이대리", "박과장", "최부장", "정사원", "강매니저"]
PRIZES = ["아이패드 Pro", "갤럭시 S24", "스타벅스 상품권 5만원", "백화점 상품권 10만원",
          "에어팟 Pro", "닌텐도 스위치", "네스프레소 머신", "다이슨 청소기"]
STOCKS = ["삼성바이오로직스", "카카오페이", "크래프톤", "하이브", "에코프로비엠", "포스코홀딩스",
          "셀트리온", "LG에너지솔루션", "현대모비스", "SK하이닉스", "카카오뱅크", "네이버"]
PRODUCTS = ["홍삼 세트", "명품 가방", "패딩 점퍼", "운동화", "화장품 세트", "건강기능식품",
            "노트북", "태블릿", "블루투스 이어폰", "스마트워치"]
PROJECTS = ["Q1 전략 기획", "신규 시스템 개발", "마케팅 캠페인", "연간 보고서",
            "고객 분석 프로젝트", "글로벌 확장 TF", "앱 리뉴얼"]
SERVICES = ["넷플릭스", "유튜브 프리미엄", "멜론", "카카오페이", "토스", "쿠팡로켓와우"]
TOPICS = ["AI/머신러닝", "스타트업", "마케팅", "개발자", "디자인", "핀테크", "이커머스"]
NEWS = [
    "GPT-5 출시 예고, AI 업계 지각변동 예상",
    "국내 스타트업 투자 동향 — {month}월 결산",
    "개발자 채용 시장 변화와 전망",
    "디자인 트렌드 {year}년 키워드 분석",
    "마케팅 자동화 도구 비교 분석",
    "블록체인 기술 실용화 사례 모음",
    "리모트워크 생산성 향상 방법론",
]
PLACES = ["서울 본사 대회의실", "강남 미팅룸 302호", "Zoom 온라인", "판교 오피스 3층"]
TASKS = ["요구사항 분석 완료", "UI 목업 작성", "API 개발", "QA 테스트", "배포 준비", "문서화"]
SCHOOLS = ["서울대학교", "연세대학교", "고려대학교", "한양대학교", "성균관대학교"]
SUBJECTS_EDU = ["컴퓨터과학개론", "데이터베이스", "알고리즘", "통계학", "경제학원론"]
EVENTS_EDU = ["취업 특강", "졸업식", "수강신청", "중간고사", "기말고사"]
EVENTS_PERSONAL = ["생일 파티", "동창회", "팀 워크숍", "송년회", "신년 모임"]

FAKE_URLS = [
    "http://bit.ly/3xK9p2m",
    "http://tinyurl.com/spam123",
    "http://go.kr-event.net/win",
    "http://loan-quick.xyz/apply",
    "http://casino-vip.top/join",
    "http://prize-win.info/claim",
]

MONTHS = list(range(1, 13))
WEEKS = list(range(1, 5))


def rand(lst):
    return random.choice(lst)


def rand_amount():
    return random.choice([50, 100, 200, 300, 500, 1000, 2000, 3000])


def rand_rate():
    return round(random.uniform(2.5, 9.9), 1)


def rand_phone():
    return f"{random.randint(100,999)}-{random.randint(1000,9999)}"


def rand_date():
    base = datetime(2024, 1, 1)
    delta = timedelta(days=random.randint(0, 365))
    return (base + delta).strftime("%Y년 %m월 %d일")


def rand_order_id():
    return f"ORD{random.randint(10000000, 99999999)}"


def rand_tracking_id():
    return f"{random.randint(100000000000, 999999999999)}"


def rand_version():
    return f"v{random.randint(1,5)}.{random.randint(0,9)}.{random.randint(0,9)}"


def rand_semester():
    return random.choice(["1", "2"])


def rand_address():
    cities = ["서울시 강남구", "서울시 마포구", "경기도 성남시", "부산시 해운대구", "인천시 연수구"]
    return rand(cities) + f" {random.randint(1,999)}번지"


def rand_amount2(base):
    return int(base * random.uniform(1.2, 2.0))


def fill_template(template: str) -> str:
    name = rand(NAMES)
    amount = rand_amount()
    amount2_val = rand_amount2(amount * 10000) // 10000
    month = rand(MONTHS)
    week = rand(WEEKS)
    company_spam = rand(COMPANIES_SPAM)
    company_ham = rand(COMPANIES_HAM)

    replacements = {
        "{name}": name,
        "{amount}": str(amount),
        "{amount2}": str(amount2_val),
        "{rate}": str(rand_rate()),
        "{phone}": rand_phone(),
        "{fake_url}": rand(FAKE_URLS),
        "{company}": company_spam,
        "{prize}": rand(PRIZES),
        "{deadline}": rand_date(),
        "{count}": str(random.randint(10, 500)),
        "{stock}": rand(STOCKS),
        "{product}": rand(PRODUCTS),
        "{project}": rand(PROJECTS),
        "{service}": rand(SERVICES),
        "{sender}": rand(SENDERS),
        "{place}": rand(PLACES),
        "{task1}": rand(TASKS),
        "{task2}": rand(TASKS),
        "{task3}": rand(TASKS),
        "{topic}": rand(TOPICS),
        "{news1}": rand(NEWS).replace("{month}", str(month)).replace("{year}", "2024"),
        "{news2}": rand(NEWS).replace("{month}", str(month)).replace("{year}", "2024"),
        "{news3}": rand(NEWS).replace("{month}", str(month)).replace("{year}", "2024"),
        "{month}": str(month),
        "{week}": str(week),
        "{time}": f"{random.randint(9,18)}:{random.choice(['00','30'])}",
        "{document}": rand(["기획서", "보고서", "제안서", "결과물", "발표자료"]),
        "{event}": rand(EVENTS_PERSONAL),
        "{school}": rand(SCHOOLS),
        "{subject}": rand(SUBJECTS_EDU),
        "{semester}": rand_semester(),
        "{order_id}": rand_order_id(),
        "{tracking_id}": rand_tracking_id(),
        "{address}": rand_address(),
        "{version}": rand_version(),
    }

    for key, val in replacements.items():
        template = template.replace(key, val)
    return template


def generate_email(label: str):
    if label == "spam":
        category = rand(list(SPAM_TEMPLATES.keys()))
        tmpl = SPAM_TEMPLATES[category]
    else:
        category = rand(list(HAM_TEMPLATES.keys()))
        tmpl = HAM_TEMPLATES[category]

    subject = fill_template(rand(tmpl["subjects"]))
    body = fill_template(rand(tmpl["bodies"]))
    return subject, body, label, category


def main():
    total = 5000
    spam_count = 2000   # 40% 스팸
    ham_count = 3000    # 60% 정상

    labels = ["spam"] * spam_count + ["ham"] * ham_count
    random.shuffle(labels)

    os.makedirs("data", exist_ok=True)
    output_path = "data/korean_spam_dataset.csv"

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "subject", "body", "label", "category"])
        for i, label in enumerate(labels, start=1):
            subject, body, lbl, category = generate_email(label)
            writer.writerow([i, subject, body, lbl, category])

    print(f"[완료] 데이터셋 생성: {output_path}")
    print(f"  전체: {total}개  |  스팸: {spam_count}개  |  정상: {ham_count}개")


if __name__ == "__main__":
    main()
