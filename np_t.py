import numpy as np
import os

# 파일 경로
input_path = "data/pp.npy"
output_txt = "image_spectra.txt"
preview_txt = "preview_spectra.txt"

# 파일 존재 확인
if not os.path.exists(input_path):
    raise FileNotFoundError(f"❌ 파일을 찾을 수 없습니다: {input_path}")

# 파일 로드
cube = np.load(input_path)  # shape: (H, W, B)
print(f"✅ 불러온 cube shape: {cube.shape}")

# (H, W, B) → (H*W, B)
H, W, B = cube.shape
pixels = cube.reshape(-1, B)

# NaN, Inf 처리
pixels = np.nan_to_num(pixels, nan=0.0, posinf=1.0, neginf=0.0)

# 전체 저장
np.savetxt(output_txt, pixels, delimiter=",", fmt="%.6f")
print(f"📁 전체 데이터 저장 완료: {output_txt} (shape: {pixels.shape})")

# preview용 상위 1000개만 저장
np.savetxt(preview_txt, pixels[:1000], delimiter=",", fmt="%.6f")
print(f"👁️ 미리보기 저장 완료: {preview_txt} (1000줄)")