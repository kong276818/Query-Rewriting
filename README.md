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

#!/usr/bin/env bash
set -euo pipefail

### ==============================================================
### 0) 새 가상환경 만들기 (conda 또는 venv)
### ==============================================================
echo "=== [0] Python 가상환경 생성 ==="
PY_VER=3.10
if command -v conda &>/dev/null; then
  conda create -n t5qr python=${PY_VER} -y
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate t5qr
else
  python3 -m venv .venv
  source .venv/bin/activate
fi

### ==============================================================
### 1) 저장소 클론 & 이동
### ==============================================================
echo "=== [1] 저장소 클론 ==="
REPO_URL="https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git"
REPO_DIR="YOUR_REPO_NAME"
if [ ! -d "$REPO_DIR" ]; then
  git clone "$REPO_URL"
fi
cd "$REPO_DIR"

### ==============================================================
### 2) PyTorch 설치
### ==============================================================
echo "=== [2] PyTorch 설치 ==="
pip install --upgrade pip
if command -v nvidia-smi &>/dev/null; then
  echo "CUDA GPU 감지됨 → GPU 버전 설치"
  pip install torch --index-url https://download.pytorch.org/whl/cu121
else
  echo "GPU 없음 → CPU 버전 설치"
  pip install torch --index-url https://download.pytorch.org/whl/cpu
fi

### ==============================================================
### 3) 나머지 의존성 설치
### ==============================================================
echo "=== [3] 필수 패키지 설치 ==="
cat > requirements.txt <<'REQ'
transformers>=4.40.0
huggingface_hub>=0.22.0
accelerate>=0.28.0
sentencepiece>=0.1.99
REQ
pip install -r requirements.txt

### ==============================================================
### 4) 설치 검증
### ==============================================================
echo "=== [4] 설치 검증 ==="
python - <<'PY'
import transformers, sentencepiece, huggingface_hub, torch
print("--- Installation Check ---")
print("transformers:", transformers.__version__)
print("sentencepiece:", sentencepiece.__version__)
print("huggingface_hub:", huggingface_hub.__version__)
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("------------------------")
PY

### ==============================================================
### 5) 빠른 시작 (Python API) 테스트
### ==============================================================
echo "=== [5] Python API 테스트 ==="
python - <<'PY'
from transformers import pipeline, set_seed
set_seed(42)

MODEL_ID = "prhegde/t5-query-reformulation-RL"  # 기본: 영어 모델
# 한국어 입력 위주라면 아래로 교체:
# MODEL_ID = "KETI-AIR/ke-t5-base"
# MODEL_ID = "paust/pko-t5-base"

rewriter = pipeline(
    "text2text-generation",
    model=MODEL_ID,
    tokenizer=MODEL_ID,
    device=0 if __import__('torch').cuda.is_available() else -1
)

def make_prompt(q: str) -> str:
    return f"아래 질의를 한국어로 더 구체적이고 검색엔진 친화적으로 재작성:\n질의: {q}\n재작성:"

queries = [
    "오늘 서울 날씨 어때",
    "노트북 싸게 사는 법 알려줘",
    "코로나 증상 뭐있어?",
    "겨울에 가볼만한 국내 여행지 추천"
]
prompts = [make_prompt(q) for q in queries]

results = rewriter(
    prompts,
    max_new_tokens=96,
    num_beams=5,
    num_return_sequences=3,
    no_repeat_ngram_size=3
)

for i, q in enumerate(queries):
    print(f"원본 쿼리: {q}")
    for j, out in enumerate(results[i]):
        print(f"  => 후보 {j+1}: {out['generated_text'].strip()}")
    print("-" * 20)
PY

### ==============================================================
### 6) CLI 사용 예시
### ==============================================================
echo "=== [6] CLI 예시 ==="
mkdir -p src
cat > src/infer.py <<'PY'
import argparse
from transformers import pipeline, set_seed
import torch

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--queries", nargs="+", required=True)
    p.add_argument("--model-id", type=str, default="prhegde/t5-query-reformulation-RL")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--num-beams", type=int, default=5)
    p.add_argument("--num-return-sequences", type=int, default=3)
    p.add_argument("--no-repeat-ngram-size", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    rewriter = pipeline(
        "text2text-generation",
        model=args.model_id,
        tokenizer=args.model_id,
        device=args.device
    )

    def make_prompt(q):
        return f"아래 질의를 한국어로 더 구체적이고 검색엔진 친화적으로 재작성:\n질의: {q}\n재작성:"

    prompts = [make_prompt(q) for q in args.queries]
    results = rewriter(
        prompts,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        num_return_sequences=args.num_return_sequences,
        no_repeat_ngram_size=args.no_repeat_ngram_size
    )
    for i, q in enumerate(args.queries):
        print(f"원본 쿼리: {q}")
        for j, out in enumerate(results[i]):
            print(f"  => 후보 {j+1}: {out['generated_text'].strip()}")

if __name__ == "__main__":
    main()
PY

echo "CLI 실행 예:"
echo "python -m src.infer --queries \"오늘 서울 날씨 어때\" \"겨울 국내 여행지 추천\" --device 0"

### ==============================================================
### 7) 오프라인/사내망 옵션 안내
### ==============================================================
echo "=== [7] 오프라인/사내망 옵션 ==="
echo "# 인터넷 환경에서:"
echo "huggingface-cli login"
echo "huggingface-cli download prhegde/t5-query-reformulation-RL --local-dir ./models/t5qr"
echo "# 코드에서:"
echo "rewriter = pipeline('text2text-generation', model='./models/t5qr', tokenizer='./models/t5qr', device=0)"

### ==============================================================
### 8) 트러블슈팅 안내
### ==============================================================
cat <<'TIPS'
[자주 겪는 이슈]
- HfHubHTTPError: huggingface_hub 업그레이드 → pip install -U huggingface_hub
- device_map="auto" 에러: accelerate 설치 필요 → pip install accelerate
- sentencepiece 관련 에러: pip install sentencepiece
- max_new_tokens/max_length 중복: 하나만 사용 (권장 max_new_tokens)
- 출력이 : 또는 의미 없음: 한국어 T5 모델로 교체 또는 번역 파이프라인
- GPU 인식 실패: CUDA/PyTorch 버전 불일치 확인
TIPS

echo "✅ 모든 설치 및 테스트 완료"
