# Query-Rewriting
Query rewriting is the process of modifying a user's original search query 


# Query-Rewriting

Query rewriting은 사용자의 원래 검색 쿼리를 수정하는 과정을 의미합니다.

---

## 📌 개요
Query rewriting(쿼리 재작성)은 검색 시스템이나 AI 기반 응답 시스템에서 매우 중요한 역할을 합니다.  
사용자의 의도를 더 정확하게 파악하고, 쿼리를 적절한 형태로 변환하여 검색 결과의 정확도와 관련성을 크게 향상시킬 수 있습니다.

---

## ✨ 주요 기능
- **자연어 기반 쿼리 정규화**
- **의도 파악을 기반으로 한 쿼리 변환**
- **멀티턴 대화 지원**
- **모듈형 구조로 손쉬운 시스템 통합**

---

## 🚀 활용 사례
- 검색엔진 고도화 및 최적화
- 챗봇 및 대화형 AI 시스템
- 전자상거래 상품 검색 정확도 향상
- RAG(Retrieval-Augmented Generation) 기반 응답 생성

---

## 🛠️ 사용 기술
- **언어**: Python
- **NLP 라이브러리**: spaCy, NLTK, Hugging Face Transformers
- **API**: RESTful API 또는 추론 엔드포인트

---

## 📂 프로젝트 구조

| 디렉토리/파일        | 설명                              |
|----------------------|-----------------------------------|
| `data/`              | 학습 및 평가용 데이터셋           |
| `models/`            | 사전학습 또는 커스텀 모델         |
| `src/`               | 핵심 코드 (전처리, 모델 호출 등) |
| `README.md`          | 프로젝트 설명 파일                |

---

## 📄 문서 기여
이 리포지토리는 지속적으로 발전 중이며, 피드백 및 기여를 환영합니다.

# T5 기반 검색 쿼리 재작성 프로젝트 (Query Rewriting with T5)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/prhegde/t5-query-reformulation-RL)

사용자의 원본 검색 쿼리를 검색 엔진에 더 친화적인 형태로 수정하여 검색 결과의 정확도와 관련성을 높이는 쿼리 재작성(Query Rewriting) 프로젝트입니다.

---

### 📌 개요

이 프로젝트는 사용자의 자연어 쿼리 의도를 더 정확하게 파악하고, 이를 검색에 최적화된 형태로 변환하는 기능을 제공합니다. `Hugging Face Transformers` 라이브러리를 기반으로 하여 강력한 사전 학습 언어 모델(T5)을 활용합니다.

### ✨ 주요 기능

-   **자연어 쿼리 정규화**: 구어체, 오탈자, 불완전한 문장 등을 명확한 쿼리로 변환합니다.
-   **의도 파악 기반 변환**: 사용자의 숨은 의도를 파악하여 더 구체적인 키워드를 포함한 쿼리를 생성합니다.
-   **모듈형 구조**: Python API 및 CLI를 제공하여 기존 시스템에 손쉽게 통합할 수 있습니다.

### 🚀 활용 사례

-   검색엔진 고도화 및 최적화
-   챗봇 및 대화형 AI 시스템의 이해 능력 강화
-   전자상거래 상품 검색 정확도 향상
-   RAG(Retrieval-Augmented Generation) 기반 응답 생성 시스템의 검색 품질 개선

---

### 📂 설치

#### 사전 요구사항
-   **Python**: 3.9 ~ 3.11 권장
-   **OS**: Windows / Linux / macOS 지원
-   **(선택)** NVIDIA GPU + CUDA가 설치된 경우 추론 속도가 크게 향상됩니다.

#### 0) 새 가상환경 만들기 (권장)
```bash
# conda 예시
conda create -n t5qr python=3.10 -y
conda activate t5qr

# 또는 venv 예시
# python -m venv .venv && source .venv/bin/activate  (Windows: .venv\Scripts\activate)
1) 저장소 클론 & 이동
Bash

# git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
# cd YOUR_REPO_NAME
2) PyTorch 설치
GPU 유무에 따라 아래 옵션 중 하나를 선택하여 설치합니다.

Bash

# pip 최신 버전으로 업그레이드
pip install --upgrade pip

# GPU (CUDA 12.1) 사용자 예시
pip install torch --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# CPU 전용 사용자
# pip install torch --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
3) 나머지 의존성 설치
requirements.txt 파일을 통해 필요한 라이브러리를 한 번에 설치합니다.

Bash

pip install -r requirements.txt
requirements.txt 예시:

Shell

transformers>=4.40.0
huggingface_hub>=0.22.0
accelerate>=0.28.0
sentencepiece>=0.1.99
4) 설치 검증
아래 명령어를 실행하여 주요 라이브러리가 올바르게 설치되었는지 확인합니다.

Bash

python - << 'PY'
import transformers, sentencepiece, huggingface_hub, torch
print("--- Installation Check ---")
print("transformers:", transformers.__version__)
print("sentencepiece:", sentencepiece.__version__)
print("huggingface_hub:", huggingface_hub.__version__)
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("------------------------")
PY
⚙️ 어떻게 동작하나 (개요)
이 리포지토리는 transformers의 text2text-generation 파이프라인을 핵심으로 사용합니다.

Hugging Face Hub에 등록된 prhegde/t5-query-reformulation-RL 모델 (T5-base 아키텍처에 RL 미세튜닝 적용)을 불러옵니다.

사용자의 짧고 모호한 질의를 입력받습니다.

"한국어로, 검색 키워드 포함"과 같은 프롬프트 힌트를 질의에 추가하여 모델이 원하는 형식의 결과를 생성하도록 유도합니다.

검색엔진이 이해하기 좋은 구체적인 쿼리로 재작성된 결과를 출력합니다.

추론 시 주요 하이퍼파라미터
max_new_tokens: 생성될 텍스트의 최대 길이 (권장, max_length와 동시 사용 금지)

num_beams: 빔 서치(Beam Search)의 빔 개수. 높을수록 정확도는 오르지만 속도는 저하됩니다.

do_sample: True로 설정 시 샘플링을 통해 더 다양한 결과를 생성합니다.

num_return_sequences: 반환할 후보 쿼리의 개수입니다.

no_repeat_ngram_size: 지정된 크기의 n-gram이 반복되는 것을 방지합니다.

device: 추론에 사용할 장치를 지정합니다. (0 for GPU, -1 for CPU)

🚀 빠른 시작 (Python API)
Python

from transformers import pipeline, set_seed

# 재현성을 위해 시드 고정
set_seed(42)

# 기본 모델은 영어권 데이터에 최적화되어 있습니다.
MODEL_ID = "prhegde/t5-query-reformulation-RL"

# (선택) 한국어 입력 위주라면 아래 한국어 T5 모델로 교체를 고려해 보세요.
# MODEL_ID = "KETI-AIR/ke-t5-base"
# MODEL_ID = "paust/pko-t5-base"

# 파이프라인 초기화
rewriter = pipeline(
    "text2text-generation",
    model=MODEL_ID,
    tokenizer=MODEL_ID,
    device=0  # GPU가 없으면 -1로 변경
)

# 더 나은 결과를 위한 프롬프트 엔지니어링 함수
def make_prompt(q: str) -> str:
    """모델에게 역할과 출력 형식을 명확히 지시하는 프롬프트를 생성합니다."""
    return f"아래 질의를 한국어로 더 구체적이고 검색엔진 친화적으로 재작성:\n질의: {q}\n재작성:"

# 재작성할 쿼리 목록
queries = [
    "오늘 서울 날씨 어때",
    "노트북 싸게 사는 법 알려줘",
    "코로나 증상 뭐있어?",
    "겨울에 가볼만한 국내 여행지 추천"
]

# 프롬프트 생성
prompts = [make_prompt(q) for q in queries]

# 쿼리 재작성 실행
# num_return_sequences를 3으로 설정했으므로 쿼리당 3개의 후보가 생성됩니다.
rewritten_results = rewriter(
    prompts,
    max_new_tokens=96,
    num_beams=5,
    num_return_sequences=3,
    no_repeat_ngram_size=3
)

# 결과 출력
for i, q in enumerate(queries):
    print(f"원본 쿼리: {q}")
    # rewriter의 출력은 2차원 리스트 형태입니다.
    # outs[i]는 i번째 쿼리에 대한 결과(후보 3개)를 담은 리스트입니다.
    for j in range(len(rewritten_results[i])):
        text = rewritten_results[i][j]["generated_text"].strip()
        print(f"  => 후보 {j+1}: {text}")
    print("-" * 20)

🧪 CLI 예시 (옵션)
src/infer.py와 같은 스크립트를 만들어두면 터미널에서 간편하게 실행할 수 있습니다.

Bash

python -m src.infer \
  --queries "오늘 서울 날씨 어때" "겨울 국내 여행지 추천" \
  --model-id "prhegde/t5-query-reformulation-RL" \
  --device 0 \
  --max-new-tokens 96 \
  --num-beams 5 \
  --num-return-sequences 3
참고: src/infer.py 파일은 위 Python API와 동일한 로직을 argparse 등으로 감싸서 구현하면 됩니다.

🔧 옵션 가이드
정밀도 vs 속도/다양성
정확도 우선: num_beams=5~8, do_sample=False (기본값)

다양성/창의성 우선: do_sample=True, top_p=0.9, temperature=0.7

한국어 입력 안정화
한국어 특화 T5 모델 사용: KETI-AIR/ke-t5-base 등으로 MODEL_ID를 교체합니다.

프롬프트 강화: 프롬프트에 "반드시 한국어로 작성해줘" 와 같은 명시적 지침을 추가합니다.

(고급) 번역 API(Papago, Google)를 사용하여 한국어 -> 영어 -> 재작성(영어모델) -> 한국어 파이프라인을 구축할 수도 있습니다 (추가 비용 및 지연 발생).

디바이스 설정
device=0: 첫 번째 GPU 사용

device=-1: CPU 사용

device_map="auto": accelerate 라이브러리를 활용하여 여러 GPU에 모델을 자동 분산 (메모리가 부족할 때 유용)

🧱 오프라인/사내망(방화벽) 환경
인터넷이 제한된 환경에서는 모델을 미리 다운로드하여 로컬에서 사용할 수 있습니다.

1) 모델 로컬에 다운로드
인터넷이 되는 환경에서 huggingface-cli를 사용하여 모델을 다운로드합니다.

Bash

# Hugging Face 계정 로그인 (필요 시)
huggingface-cli login

# 지정된 로컬 디렉토리로 모델 스냅샷 다운로드
huggingface-cli download prhegde/t5-query-reformulation-RL --local-dir ./models/t5qr
2) 로컬 경로로 모델 로드
Python 코드에서 model과 tokenizer 인자에 온라인 ID 대신 로컬 경로를 지정합니다.

Python

rewriter = pipeline(
    "text2text-generation",
    model="./models/t5qr",      # 로컬 경로 지정
    tokenizer="./models/t5qr",  # 로컬 경로 지정
    device=0
)
프록시 설정
프록시 서버가 필요한 환경에서는 환경 변수를 설정하세요.

Bash

# Windows
set HTTP_PROXY=http://user:pass@host:port
set HTTPS_PROXY=http://user:pass@host:port

# Linux/macOS
export HTTP_PROXY="http://user:pass@host:port"
export HTTPS_PROXY="http://user:pass@host:port"
🩹 트러블슈팅 (자주 겪는 이슈)
증상 / 오류 메시지	원인	해결책
HfHubHTTPError import 에러	huggingface_hub 라이브러리 버전이 낮음.	pip install -U huggingface_hub 로 업그레이드.
Using a device_map ... requires accelerate	device_map="auto" 사용 시 accelerate 미설치	pip install accelerate 를 실행하거나 device=0 또는 device=-1 로 변경.
Cannot instantiate ... sentencepiece	sentencepiece 라이브러리 미설치	pip install sentencepiece
Both max_new_tokens and max_length ... 경고	길이 관련 파라미터 중복 사용	둘 중 하나만 사용합니다. (max_new_tokens 권장)
TypeError: list indices must be integers or slices	배치 처리 + num_return_sequences>1 출력 구조 오해	outs[i][j]["generated_text"] 와 같이 2차원 리스트로 인덱싱합니다.
출력이 : 또는 의미 없는 영어만 나옴	영어 학습 모델에 한국어 프롬프트가 바로 입력됨	한국어 T5 모델로 교체하거나, 프롬프트에 "한국어로" 지침을 명확히 합니다.
GPU를 찾지 못하거나 CUDA 관련 에러 발생	설치된 PyTorch와 시스템의 CUDA 버전이 불일치	시스템의 CUDA 버전을 확인하고, 그에 맞는 PyTorch를 공식 홈페이지에서 찾아 재설치합니다.

Sheets로 내보내기
🔁 재현성 & 성능 팁
결과 고정: set_seed(정수)를 코드 상단에 추가하여 매번 동일한 결과를 얻을 수 있습니다.

출력 길이 제어: 너무 긴 출력을 방지하려면 max_new_tokens를 64~128 사이로 조정하세요.

반복 방지: no_repeat_ngram_size=3은 같은 구절이 3번 이상 반복되는 것을 막아주는 효과적인 옵션입니다.

메모리 부족: Out-of-Memory 에러 발생 시 num_beams, num_return_sequences 값을 낮추거나, 배치 크기를 줄이세요.

📄 참고 자료
Model Card: prhegde/t5-query-reformulation-RL on Hugging Face

Related Research:

Nogueira & Cho (2017). Task-Oriented Query Reformulation with Reinforcement Learning.

QuAC (2018), CANARD (2019), QReCC (2020) Datasets

Lin et al. (2022). CONQRR: A Conversational Query Reformulation and Reranking Framework.

Hedge et al. (2023). RLQR: Reinforcement Learning for Query Reformulation.
