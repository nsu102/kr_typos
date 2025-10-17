# Korean Typo Generator & G-Eval Code-Switching Evaluator

한글 오타 생성 및 코드 스위칭 응답 평가를 위한 연구 프로젝트

## 프로젝트 개요

이 프로젝트는 두 가지 주요 기능을 제공합니다:

1. **한글 오타 생성기** (`make_typos.py`): 한글 문장에 대해 5가지 유형의 체계적인 오타를 생성
2. **G-Eval 평가기** (`g-eval-cs.py`): 코드 스위칭 질문에 대한 언어 모델 응답을 G-Eval 방법론으로 평가

## 주요 기능

### 1. 한글 오타 생성 (`make_typos.py`)

#### 오타 유형

**1. 교체 (Substitution)** - 유사 자모로 교체
- 유사 자모 교체 (예: ㄱ → ㄲ, ㅋ)
- 키보드 인접 자모 교체 (두벌식 기준)
- 음운적 유사 교체 (예: 지 → 치)

**2. 삭제 (Deletion)** - 자모 또는 음절 삭제
- 자모 삭제 (예: 국 → 구)
- 음절 삭제

**3. 추가 (Insertion)** - 자모 추가
- 종성 추가 (예: 가 → 각)
- 자모 중복

**4. 전치 (Transposition)** - 자모 순서 바꾸기
- 초성-중성 전치 (예: 호 → ㅗㅎ)

**5. 띄어쓰기 오류 (Spacing)**
- 공백 제거 (예: 한국 어 → 한국어)
- 공백 추가 (예: 한국어 → 한국 어)
- 자모 사이 공백 (예: 국 → 구 ㄱ, ㄱ ㅜㄱ)

#### 사용법

```bash
python make_typos.py --input data/mkqa_kr.json --output data/typos_data.json
```

**입력 형식**: 한글 문장 리스트 (JSON)
```json
["한글 문장", "또 다른 문장", "세 번째 문장"]
```

**출력 형식**: 각 오타 유형별 1개/2개 오류 버전
```json
[
  {
    "id": 1,
    "original": "원본 문장",
    "substitution": {
      "1_error": {"text": "오타 1", "errors": ["장 -> 창"]},
      "2_errors": {"text": "오타 2", "errors": ["장 -> 창", "문 -> 운"]}
    },
    "deletion": {...},
    "insertion": {...},
    "transposition": {...},
    "spacing": {...}
  }
]
```

### 2. G-Eval 평가 (`g-eval-cs.py`)

G-Eval 방법론을 사용하여 코드 스위칭 질문에 대한 언어 모델 응답을 평가합니다.

#### 평가 기준 (4가지 메트릭)

1. **Helpfulness (유용성)** - 사용자 요구를 효과적으로 충족 (1-10)
2. **Relevance (관련성)** - 주제와의 연관성 유지 (1-10)
3. **Accuracy (정확성)** - 정보의 사실적 정확성 (1-10)
4. **Depth (깊이)** - 응답의 상세함과 포괄성 (1-10)

**최종 점수** = 4가지 메트릭의 평균 (1-10 척도)

#### 평가 방식

- **5C2 방식**: 5개 Case 중 2개를 선택하여 쌍별 비교 (총 10개 조합)
- **평가 모델**: qwen_72b, qwen_7b 각각 독립 평가
- **멀티스레딩**: 병렬 처리로 빠른 평가
- **체크포인트**: 진행 상황 자동 저장 및 복구

#### 사용법

```bash
python g-eval-cs.py \
  --input_file data/response_data_typo.json \
  --output_file data/g_eval_results.json \
  --model openai/gpt-4o-mini \
  --workers 3 \
  --checkpoint_every 50
```

**주요 옵션**:
- `--input_file`: 입력 데이터 파일 경로
- `--output_file`: 출력 결과 파일 경로
- `--model`: 평가용 LLM 모델 (기본값: gpt-4o-mini)
- `--workers`: 멀티스레딩 워커 수 (기본값: 3)
- `--limit`: 평가할 항목 수 제한 (선택사항)
- `--checkpoint_every`: 체크포인트 저장 주기 (기본값: 50)

## 프로젝트 구조

```
kr_typos/
├── make_typos.py           # 한글 오타 생성기
├── g-eval-cs.py           # G-Eval 평가기
├── data/
│   ├── mkqa_kr.json       # 입력 한글 문장 데이터 (120KB)
│   └── typos_data.json    # 생성된 오타 데이터 (3.5MB)
└── README.md
```

## 데이터 파일 정보

| 파일 | 크기 | 설명 |
|------|------|------|
| mkqa_kr.json | 120KB | MKQA 데이터셋의 한글 문장 |
| typos_data.json | 3.5MB | 5가지 오타 유형 × 2개 오류 버전 |

## 환경 설정

### 필수 패키지

```bash
pip install openai tqdm
```

### 환경 변수

G-Eval 평가기 사용 시 OpenRouter API 키 필요:

```bash
export OPENROUTER_API_KEY="your_api_key_here"
```

## 기술적 세부사항

### 오타 생성기

- **한글 분해/조합**: 유니코드 기반 초성/중성/종성 처리
- **위치 추적**: 중복 오류 방지를 위한 사용된 위치 추적
- **강제 교체**: 모든 경우에 반드시 오타 생성 보장
- **유사 자모**: 발음 유사성 기반 매핑
- **키보드 매핑**: 두벌식 자판 기준 인접 자모

### G-Eval 평가기

- **4가지 핵심 메트릭**: 유용성, 관련성, 정확성, 깊이
- **Rate limit 처리**: 지수 백오프 재시도 로직
- **체크포인트**: 중단 후 이어서 실행 가능
- **멀티스레딩**: 병렬 평가로 속도 향상
- **통계 분석**: 모델별, 메트릭별 상세 분석

## 사용 예시

### 1. 오타 데이터 생성

```bash
python make_typos.py \
  --input data/mkqa_kr.json \
  --output data/typos_data.json
```

### 2. G-Eval 평가 실행

```bash
python g-eval-cs.py \
  --input_file data/code_switched_response.json \
  --output_file data/g_eval_results.json \
  --workers 5 \
  --checkpoint_every 50
```

### 3. 평가 결과 확인

평가 완료 시 자동으로 통계 출력:
- 모델별 평균 점수
- 메트릭별 평균 점수
- Case 간 점수 차이
- 종합 분석

## 연구 질문

이 프로젝트는 다음 연구 질문에 답하기 위해 설계되었습니다:

1. 오타가 있는 코드 스위칭 입력에 대해 언어 모델이 얼마나 강건한가?
2. 오타 유형에 따라 모델 성능이 어떻게 달라지는가?
3. 오타 개수가 증가할수록 응답 품질이 얼마나 저하되는가?
4. 모델 크기(72B vs 7B)가 오타 처리 능력에 영향을 주는가?

## 주의사항

1. **API 키**: G-Eval 평가 시 OpenRouter API 키 필수
2. **Rate Limit**: API 호출 제한 고려 (자동 재시도 포함)
3. **메모리**: 대용량 데이터 처리 시 메모리 사용량 확인
4. **체크포인트**: 장시간 실행 시 체크포인트 기능 활용 권장

## 라이선스

MIT License

## 기여

이슈 및 풀 리퀘스트 환영합니다.

## 문의

프로젝트 관련 문의사항은 이슈를 통해 남겨주세요.
