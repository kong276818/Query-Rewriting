import os
from transformers import pipeline, set_seed

# --- HF Hub import (호환성 보장) ------------------------------
from huggingface_hub import snapshot_download  # login은 꼭 필요할 때만
try:
    # 최신 버전: 여기서 제공됨
    from huggingface_hub.utils import HfHubHTTPError
except Exception:
    # 구버전 호환: 동일 이름의 예외 클래스를 임시 정의
    class HfHubHTTPError(Exception):
        pass

MODEL_ID = "prhegde/t5-query-reformulation-RL"
FALLBACKS = [
    "KETI-AIR/ke-t5-base",   # 한국어 T5
    "google/flan-t5-base"    # 범용 대안
]

def ensure_local(model_id, local_dir):
    try:
        # 오프라인/방화벽 대비: 모델을 디스크로 스냅샷
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        return local_dir
    except HfHubHTTPError as e:
        print(f"[WARN] 다운로드 실패(HfHubHTTPError): {e}")
        return None
    except Exception as e:
        print(f"[WARN] 기타 다운로드 에러: {e}")
        return None

def _try_make_pipeline(model, tokenizer):
    """
    accelerate 미설치/환경 문제 대비:
    1) device_map='auto' 시도
    2) 실패하면 device 인덱스 기반으로 재시도
    """
    # 1) device_map='auto' (가능하면 FP16 자동)
    try:
        return pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
            torch_dtype="auto"
        )
    except Exception as e:
        print(f"[INFO] device_map='auto' 실패 → device 폴백 사용: {e}")

    # 2) device 인덱스 폴백
    try:
        import torch
        device = 0 if torch.cuda.is_available() else -1  # -1: CPU
    except Exception:
        device = -1

    return pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device
    )

def load_pipeline():
    set_seed(42)

    # 1) 우선 원 모델(온라인/캐시) 시도
    try:
        print(f"[INFO] 1차 로드: {MODEL_ID}")
        return _try_make_pipeline(MODEL_ID, MODEL_ID)
    except Exception as e:
        print(f"[WARN] 온라인 로드 실패: {e}")

    # 2) 스냅샷 다운로드 후 로컬 경로 로드
    local_dir = ensure_local(MODEL_ID, "./models/prhegde_t5_qr")
    if local_dir:
        try:
            print(f"[INFO] 2차 로드(로컬): {local_dir}")
            return _try_make_pipeline(local_dir, local_dir)
        except Exception as e:
            print(f"[WARN] 로컬 로드 실패: {e}")

    # 3) 폴백들 순차 시도
    for fb in FALLBACKS:
        try:
            print(f"[INFO] 폴백 시도: {fb}")
            return _try_make_pipeline(fb, fb)
        except Exception as e:
            print(f"[WARN] 폴백 실패({fb}): {e}")

    raise RuntimeError("모델 로딩에 모두 실패했습니다. 네트워크/프록시/토큰/버전 확인 필요.")

rewriter = load_pipeline()

def make_prompt(q: str) -> str:
    return (
        "아래 질의를 한국어로 더 구체적이고 검색엔진 친화적으로 재작성:\n"
        f"질의: {q}\n재작성:"
    )

original_queries = [
    "오늘 서울 날씨 어때",
    "노트북 싸게 사는 법 알려줘",
    "코로나 증상 뭐있어?",
    "겨울에 가볼만한 국내 여행지 추천"
]
prompts = [make_prompt(q) for q in original_queries]

outs = rewriter(
    prompts,
    max_length=96,
    num_beams=5,
    num_return_sequences=3,
    early_stopping=True,
    no_repeat_ngram_size=3
)

for i, q in enumerate(original_queries):
    print(f"원본: {q}")
    for j in range(3):
        print("  =>", outs[i*3 + j]["generated_text"].strip())
