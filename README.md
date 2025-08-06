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
#!/bin/bash

# --- 예시 1: 영어-한국어 번역 ---
# 'translate English to Korean: ' 이라는 접두사를 사용하여 번역 작업을 지시합니다.
echo "--- 1. 영어-한국어 번역 요청 ---"
curl -X POST \
     -H "Content-Type: application/json" \
     -d '{"input_text": "translate English to Korean: \"Hello, world\""}' \
     https://api.example-t5-provider.com/v1/generate

# 예상되는 API 응답 (JSON 형식):
# {
#   "output_text": "안녕하세요"
# }

echo -e "\n\n" # 줄바꿈

# --- 예시 2: 긴 텍스트 요약 ---
# 'summarize: ' 접두사를 사용하여 요약 작업을 지시합니다.
echo "--- 2. 뉴스 기사 요약 요청 ---"
curl -X POST \
     -H "Content-Type: application/json" \
     -d '{"input_text": "summarize: \"The James Webb Space Telescope (JWST) is a space telescope designed primarily to conduct infrared astronomy. As the largest optical telescope in space, its high resolution and sensitivity allow it to view objects too old, distant, or faint for the Hubble Space Telescope.\""}' \
     https://api.example-t5-provider.com/v1/generate

# 예상되는 API 응답 (JSON 형식):
# {
#   "output_text": "제임스 웹 우주 망원경은 적외선 천문학을 위한 우주 망원경으로, 허블 망원경보다 더 멀고 오래된 천체를 관측할 수 있습니다."
# }

echo -e "\n\n" # 줄바꿈

# --- 예시 3: 문법 교정 ---
# 여기서는 별도의 접두사 없이 문맥만으로 작업을 유추하게 하거나,
# 'fix grammar: ' 또는 'correct this: ' 같은 접두사를 사용할 수 있습니다.
echo "--- 3. 문법 교정 요청 ---"
curl -X POST \
     -H "Content-Type: application/json" \
     -d '{"input_text": "correct this: \"She no went to the store.\""}' \
     https://api.example-t5-provider.com/v1/generate

# 예상되는 API 응답 (JSON 형식):
# {
#   "output_text": "She did not go to the store."
# }

echo -e "\n"

이처럼 어떤 작업이든 **'명령 텍스트'**를 입력하면 **'결과 텍스트'**를 출력하는 만능(all-in-one) 프레임워크를 사용하는 것이 T5의 핵심입니다.

T5의 구조와 학습
기반 기술: T5는 트랜스포머(Transformer) 라는, 현재 대부분의 AI 언어 모델(GPT 등)이 사용하는 검증된 아키텍처를 기반으로 합니다.

사전 학습 (Pre-training): 구글은 C4 (Colossal Clean Crawled Corpus) 라는 인터넷에서 수집한 방대한 양의 텍스트 데이터를 T5에게 학습시켰습니다. 이 과정에서 T5는 문법, 상식, 추론 능력 등 언어에 대한 전반적인 이해도를 갖추게 됩니다.

미세 조정 (Fine-tuning): 사전 학습된 T5를 우리가 원하는 특정 작업(예: 고객 문의 답변, 법률 문서 요약 등)에 맞게 추가로 학습시켜 성능을 극대화할 수 있습니다.

결론적으로 T5는,

하나의 모델로 번역, 요약, 분류, 질의응답 등 다양한 자연어 처리(NLP) 작업을 수행할 수 있도록 설계된, 매우 유연하고 강력한 구글의 AI 언어 모델입니다.
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

