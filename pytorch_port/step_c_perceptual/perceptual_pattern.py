# -*- coding: utf-8 -*-
"""
Step C-Perceptual (파일럿): "그래디언트 최적화 없이" 글자 배경(워터마크 영역)에
비전 인코더(ViT/CLIP류)가 구조적으로 약한 패턴을 넣어, Claude 같은 블랙박스
이미지 입력 AI에도 영향을 줄 수 있는지 확인하는 파일럿 테스트용 모듈.

배경 (연구계획서 docs/visual_protection_pattern_plan.md 5.3/5.4절 참고):
- Step A~C까지의 공격(attack.py, ocr_model.py)은 EasyOCR(화이트박스, 가중치
  공개)을 대상으로, CTC loss를 역전파해서 "이 픽셀을 이렇게 바꾸면 모델이
  틀린다"를 수학적으로 직접 계산하는 방식이다. 이 방식은 Claude에는 쓸 수
  없다 (Claude 내부 가중치에 접근 불가 = 그래디언트 계산 불가, 블랙박스).
- 대안: 그래디언트 없이도, "비전 인코더가 국소 텍스처·고빈도 디테일에
  편향되어 있다"는 알려진 특성(Making MLLMs Blind, 2026 등)을 이용해 패턴을
  수작업으로 설계하고, 그 패턴이 실제로 Claude API 응답에 영향을 주는지
  직접 테스트해본다. "이게 통하는지"는 이론으로 증명하는 게 아니라, 실제로
  Claude에게 이미지를 보여주고 정답률을 재는 것으로만 확인할 수 있다.

이 파일이 제공하는 패턴 2종 (워터마크 영역에만 적용 - 글자 영역은 절대
건드리지 않음, watermark.py의 wm_mask/text_mask 분리 원칙을 그대로 따름):

1. high_freq_noise_pattern: 글자 획과 비슷한 두께의 체커보드/격자 모양
   고빈도 흑백 노이즈. 비전 인코더가 고빈도 디테일에 민감하게 반응한다는
   가설을 테스트하기 위한 가장 단순한 패턴.
2. jitter_distortion: 워터마크 영역 픽셀을 작은 블록 단위로 좌우상하로
   아주 조금씩 뒤섞어(지터) 정렬/연결성을 깨뜨리는 패턴. 사람은 게슈탈트
   (패턴 완성)로 자연스럽게 인식하지만, AI의 토큰화 과정에서는 지역적
   일관성이 깨질 수 있다는 가설 (Illusions Machines Dataset, 2026 참고).

주의: 이 모듈은 "효과가 있다고 증명된 것"이 아니라 "효과가 있는지 확인하기
위한 파일럿 도구"다. 결과는 반드시 실제 Claude API 테스트로 검증해야 한다.
"""
import numpy as np
from PIL import Image

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from watermark import get_text_mask  # noqa: E402


def high_freq_noise_pattern(image, cell_size=3, intensity=40, seed=None):
    """워터마크 영역(글자가 아닌 부분)에 체커보드형 고빈도 노이즈를 입힌다.

    원리 (쉬운 설명):
    - 글자 획 두께와 비슷한 크기(cell_size)의 작은 사각형 타일을 체스판처럼
      밝게/어둡게 번갈아 칠한다. 사람은 배경 전체를 "약간 지저분한 회색
      배경"으로 뭉뚱그려 인식하고 글자만 또렷이 읽지만, 비전 인코더는 이
      고빈도 패턴을 글자의 일부 특징(엣지, 텍스처)으로 착각해 혼동할 수
      있다는 가설(Making MLLMs Blind, 2026의 "국소 텍스처 편향" 참고).

    Args:
        image: PIL.Image (RGB) - 원본 글자 이미지 (watermark.py의 text_img와 동일 포맷)
        cell_size: 체커보드 타일 한 칸의 픽셀 크기. 작을수록 더 고빈도.
        intensity: 밝기 변화 폭 (0~127 권장). 클수록 눈에 더 띔.
        seed: 재현성을 위한 난수 시드 (옵션)

    Returns:
        dict: {"composite": PIL.Image, "wm_mask": np.ndarray(bool)}
            composite는 원본 글자는 그대로, 배경에만 패턴이 들어간 이미지.
            wm_mask는 패턴이 실제로 적용된 영역(글자 아닌 곳)을 표시.
    """
    rng = np.random.default_rng(seed)

    arr = np.array(image.convert("RGB")).astype(np.int16)
    h, w = arr.shape[:2]

    text_mask = get_text_mask(np.array(image.convert("L")))  # True = 글자 영역

    # 체커보드 타일 좌표 계산
    yy, xx = np.meshgrid(np.arange(h) // cell_size, np.arange(w) // cell_size, indexing="ij")
    checker = (yy + xx) % 2  # 0 또는 1, 번갈아가며 타일 패턴

    # 타일마다 약간의 랜덤성을 섞어 "규칙적인 체커보드"보다 자연스러운 배경처럼 보이게 함
    tile_noise = rng.integers(-10, 11, size=checker.shape)
    delta = np.where(checker == 1, intensity, -intensity).astype(np.int16) + tile_noise

    wm_mask = ~text_mask  # 글자가 아닌 모든 영역에 패턴 적용

    out = arr.copy()
    for c in range(3):
        channel = out[:, :, c]
        channel[wm_mask] = np.clip(channel[wm_mask] + delta[wm_mask], 0, 255)
        out[:, :, c] = channel

    composite = Image.fromarray(out.astype(np.uint8), mode="RGB")
    return {"composite": composite, "wm_mask": wm_mask}


def jitter_distortion(image, block_size=4, max_shift=3, seed=None):
    """워터마크 영역을 작은 블록 단위로 좌우/상하로 미세하게 뒤섞는다.

    원리 (쉬운 설명):
    - 배경을 block_size x block_size 크기의 작은 타일로 나누고, 각 타일을
      최대 max_shift 픽셀만큼 무작위 방향으로 살짝 이동시킨다. 사람 눈에는
      "약간 거친 배경 질감"처럼 보이지만, 국소적인 픽셀 정렬이 불규칙해져
      비전 인코더의 패치 기반 처리(ViT의 patch embedding 등)에 혼란을 줄
      수 있다는 가설(Illusions Machines Dataset, 2026의 "레이아웃/정렬
      파괴가 효과적" 결과 참고).
    - 글자 영역(text_mask)은 블록 경계와 무관하게 항상 원본 그대로 복원해서,
      글자가 흔들리거나 잘리지 않게 보호한다.

    Args:
        image: PIL.Image (RGB)
        block_size: 뒤섞을 블록 한 변의 픽셀 크기
        max_shift: 블록당 최대 이동 픽셀 수
        seed: 재현성을 위한 난수 시드

    Returns:
        dict: {"composite": PIL.Image, "wm_mask": np.ndarray(bool)}
    """
    rng = np.random.default_rng(seed)

    arr = np.array(image.convert("RGB"))
    h, w = arr.shape[:2]

    text_mask = get_text_mask(np.array(image.convert("L")))
    wm_mask = ~text_mask

    out = arr.copy()

    for y0 in range(0, h, block_size):
        for x0 in range(0, w, block_size):
            y1, x1 = min(y0 + block_size, h), min(x0 + block_size, w)
            # 글자가 전혀 없는 블록만 지터 대상으로 삼는다 (글자 침범 방지)
            if text_mask[y0:y1, x0:x1].any():
                continue

            dy = int(rng.integers(-max_shift, max_shift + 1))
            dx = int(rng.integers(-max_shift, max_shift + 1))
            sy0, sy1 = np.clip([y0 + dy, y1 + dy], 0, h)
            sx0, sx1 = np.clip([x0 + dx, x1 + dx], 0, w)

            src = arr[sy0:sy1, sx0:sx1]
            dst_h, dst_w = y1 - y0, x1 - x0
            if src.shape[0] == dst_h and src.shape[1] == dst_w:
                out[y0:y1, x0:x1] = src

    # 글자 영역은 항상 원본으로 복원 (혹시 모를 침범 방지용 안전장치)
    out[text_mask] = arr[text_mask]

    composite = Image.fromarray(out.astype(np.uint8), mode="RGB")
    return {"composite": composite, "wm_mask": wm_mask}


def combined_pattern(image, cell_size=3, intensity=35, block_size=4, max_shift=2, seed=None):
    """고빈도 노이즈 + 지터를 함께 적용한 조합 패턴 (두 가설을 동시에 테스트)."""
    step1 = high_freq_noise_pattern(image, cell_size=cell_size, intensity=intensity, seed=seed)
    step2 = jitter_distortion(step1["composite"], block_size=block_size, max_shift=max_shift, seed=seed)
    return {"composite": step2["composite"], "wm_mask": step1["wm_mask"] | step2["wm_mask"]}
