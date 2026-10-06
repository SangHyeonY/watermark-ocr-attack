# -*- coding: utf-8 -*-
"""
워터마크 합성 + 마스크 생성 (FAWA 원본 로직의 PyTorch/PIL 포팅)

원본 참고: wm_grad.py (Fast-Adversarial-Watermark-Attack-on-OCR, TensorFlow 버전)

핵심 아이디어 (원본과 동일):
1. 워터마크로 쓸 글자(혹은 문양)를 기울여서 회색 텍스트 이미지로 렌더링한다.
2. 원본 글자 이미지 뒤에 이 워터마크를 합성한다 (워터마크가 배경처럼 깔림).
3. 합성 결과에서 두 개의 마스크를 분리해서 기록한다.
   - wm_mask: 워터마크 패턴이 차지하는 픽셀 영역 (True/False)
   - text_mask: 실제 글자(사람이 읽어야 하는 정보)가 차지하는 픽셀 영역
4. 공격(적대적 섭동) 시에는 wm_mask 영역에만 변화를 주도록 제한한다.
   -> 실제 글자는 전혀 건드리지 않으므로 사람의 가독성은 유지되고,
      "워터마크처럼 보이는 무늬"만 미세하게 바뀌어서 OCR을 속인다.

원본과 달리 변경한 점:
- trdg(TextRecognitionDataGenerator) 라이브러리를 사용하지 않는다.
  스모크 테스트 단계에서 trdg가 한글 렌더링을 깨뜨리는 문제가 있었기 때문에
  (WORK_LOG.md 3단계 참고), 여기서는 PIL(ImageDraw, ImageFont)로 직접
  텍스트 이미지를 렌더링한다. 이는 한글/영어 모두에 안전하게 동작한다.
"""
import math
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 기본 한글 지원 폰트 경로 (Windows 기본 설치 폰트를 WSL에서 사용)
DEFAULT_FONT_CANDIDATES = [
    "/mnt/c/Windows/Fonts/malgun.ttf",
    "/mnt/c/Windows/Fonts/NanumGothic.ttf",
]


def resolve_font(font_path=None):
    """사용할 폰트 경로를 결정한다. 지정하지 않으면 기본 후보 중 존재하는 것을 사용."""
    if font_path is not None:
        if not os.path.exists(font_path):
            raise FileNotFoundError(f"지정한 폰트가 존재하지 않습니다: {font_path}")
        return font_path
    for path in DEFAULT_FONT_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"사용 가능한 기본 폰트를 찾지 못했습니다. 후보: {DEFAULT_FONT_CANDIDATES}"
    )


def render_text_image(text, font_path=None, font_size=60, margin=20,
                       text_color=(0, 0, 0), bg_color=(255, 255, 255)):
    """텍스트를 흰 배경에 지정한 색으로 렌더링한 RGB 이미지를 반환한다.

    원본 FAWA의 '원본 글자 이미지'(input_img)에 대응.
    """
    font_path = resolve_font(font_path)
    font = ImageFont.truetype(font_path, font_size)
    bbox = font.getbbox(text)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    img_w, img_h = text_w + 2 * margin, text_h + 2 * margin
    img = Image.new("RGB", (img_w, img_h), color=bg_color)
    draw = ImageDraw.Draw(img)
    draw.text((margin - bbox[0], margin - bbox[1]), text, fill=text_color, font=font)
    return img


def generate_watermark_pattern(wm_text, size, font_path=None, font_size=100,
                                angle_deg=10, gray_value=220,
                                morph_kernel_size=5):
    """기울어진 회색 워터마크 텍스트 이미지를 생성한다.

    원본 FAWA의 gen_wm() + dilate/erode 정제 로직에 대응.

    Args:
        wm_text: 워터마크로 사용할 문자열 (예: "SAMPLE", 보호 패턴 문양 텍스트 등)
        size: 최종 캔버스 크기 (width, height) - 합성 대상 글자 이미지와 맞춘다.
        font_path: 워터마크용 폰트 경로 (None이면 기본 한글 폰트 사용)
        font_size: 워터마크 글자 크기
        angle_deg: 기울기 각도 (도 단위, 원본은 10도)
        gray_value: 워터마크 글자의 회색조 값 (0~255, 클수록 연함). 원본 FAWA는 174를
            사용했으나(76~226 범위에서 공격 시 색 변화를 줘도 자연스러운 워터마크처럼
            보이게 하려는 의도), EasyOCR(검출+인식 구조)에서는 174처럼 진하면
            워터마크가 별도 글자로 검출되는 문제가 있어 기본값을 220으로 높였다.
        morph_kernel_size: dilate/erode로 글자 테두리를 정리할 커널 크기

    Returns:
        PIL.Image (RGB) - 흰 배경에 회색 워터마크 글자가 기울어진 상태로 그려짐.
        배경은 완전히 흰색(255,255,255)이라, 이후 합성 시 배경 부분은
        영향을 주지 않는다.
    """
    font_path = resolve_font(font_path)
    font = ImageFont.truetype(font_path, font_size)

    # 1) 검은 글자로 먼저 넉넉한 캔버스에 렌더링 (회전 여유 공간 포함)
    bbox = font.getbbox(wm_text)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = max(text_w, text_h)  # 회전 시 잘리지 않도록 여유
    canvas_w, canvas_h = text_w + 2 * pad, text_h + 2 * pad
    raw = Image.new("L", (canvas_w, canvas_h), color=255)
    draw = ImageDraw.Draw(raw)
    draw.text((pad - bbox[0], pad - bbox[1]), wm_text, fill=0, font=font)

    # 2) 회전 (기울이기)
    rotated = raw.rotate(angle_deg, resample=Image.BICUBIC, fillcolor=255, expand=True)

    # 3) dilate + erode로 글자 테두리를 정리 (원본과 동일한 전처리)
    arr = np.array(rotated)
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    arr = cv2.dilate(arr, kernel, iterations=2)
    arr = cv2.erode(arr, kernel, iterations=2)

    # 3.5) 회전으로 생긴 여백을 제거하고 실제 글자가 있는 영역만 crop
    #      (이 단계가 없으면 패딩이 그대로 남아 리사이즈 후 글자가 지나치게
    #      작아지는 문제가 있었음 - 최초 구현에서 wm_mask 비율이 1% 수준으로
    #      나온 원인이 바로 이것이었다. WORK_LOG.md 4단계 참고)
    non_bg_rows = np.where((arr != 255).any(axis=1))[0]
    non_bg_cols = np.where((arr != 255).any(axis=0))[0]
    if len(non_bg_rows) > 0 and len(non_bg_cols) > 0:
        r0, r1 = non_bg_rows.min(), non_bg_rows.max() + 1
        c0, c1 = non_bg_cols.min(), non_bg_cols.max() + 1
        arr = arr[r0:r1, c0:c1]

    # 4) 배경(255)은 그대로, 글자 부분(255가 아닌 곳)만 지정한 gray_value로 채움
    bg_mask = arr == 255
    gray_arr = np.full_like(arr, gray_value)
    gray_arr[bg_mask] = 255

    wm_img_cropped = Image.fromarray(gray_arr).convert("RGB")

    # 5) crop된 워터마크를 최종 캔버스 크기의 흰 배경 중앙에 배치한다.
    #    (리사이즈로 늘려서 왜곡시키지 않고, 비율을 유지한 채 캔버스에 맞춤)
    target_w, target_h = size
    crop_w, crop_h = wm_img_cropped.size
    # 캔버스를 최대한 채우도록 비율 유지 스케일링 (스케일 업도 허용해 글자가 너무 작지 않게 함)
    scale = min(target_w / crop_w, target_h / crop_h) if crop_w > 0 and crop_h > 0 else 1.0
    new_w, new_h = max(1, int(crop_w * scale)), max(1, int(crop_h * scale))
    wm_resized = wm_img_cropped.resize((new_w, new_h), Image.BICUBIC)

    wm_img = Image.new("RGB", size, color=(255, 255, 255))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    wm_img.paste(wm_resized, box=(paste_x, paste_y))
    return wm_img


def get_text_mask(img_array, threshold_ratio=1 / 1.25):
    """글자(어두운 픽셀) 영역을 True로 표시하는 불리언 마스크를 반환한다.

    원본 FAWA의 get_text_mask()에 대응. 밝기가 threshold 이하인 픽셀을
    '글자'로 간주한다 (어두운 픽셀 = 글자, 밝은 픽셀 = 배경/워터마크 여백).
    """
    arr = np.asarray(img_array)
    if arr.ndim == 3:
        arr = np.array(Image.fromarray(arr).convert("L"))
    if arr.max() <= 1.0:
        threshold = threshold_ratio
    else:
        threshold = 255 * threshold_ratio
    return arr < threshold


def composite_text_with_watermark(text_img, wm_text="PROTECTED", font_path=None,
                                   wm_font_size=None, angle_deg=10, gray_value=220,
                                   x_shift=10):
    """원본 글자 이미지 뒤에 워터마크를 합성하고, wm_mask/text_mask를 반환한다.

    원본 FAWA wm_grad.py의 메인 합성 루프(각 input_img에 대한 for 루프)에 대응.

    Args:
        text_img: PIL.Image (RGB) - 원본 글자 이미지 (흰 배경, 검은 글자)
        wm_text: 워터마크로 쓸 문자열
        font_path: 워터마크용 폰트 경로
        wm_font_size: 워터마크 글자 크기 (None이면 text_img 높이에 비례해 자동 설정)
        angle_deg: 워터마크 기울기 각도
        gray_value: 워터마크 회색조 값. 원본 FAWA는 174를 사용했으나, 본 포팅에서
            사용하는 EasyOCR은 검출(CRAFT)+인식 구조라서 글자가 너무 또렷하면
            (낮은 gray_value일수록 진함) 워터마크 자체가 별도의 글자로 검출되어
            OCR 결과에 섞여 들어가는 문제가 있었다 (예: "학교" -> "학교EI").
            220 이상으로 연하게 설정해야 이 문제가 사라진다
            (WORK_LOG.md 4단계 "발견한 문제 2" 참고).
        x_shift: 워터마크 가로 위치 오프셋 (원본의 right_shift=10에 대응)

    Returns:
        dict with keys:
            "composite": PIL.Image (RGB) - 최종 합성 이미지 (워터마크 + 원본 글자)
            "wm_mask": np.ndarray(bool) - 워터마크 영역 마스크 (H, W)
            "text_mask": np.ndarray(bool) - 원본 글자 영역 마스크 (H, W)
            "wm_pattern": PIL.Image (RGB) - 합성 전 워터마크 패턴 자체 (디버깅/시각화용)
    """
    size = text_img.size  # (W, H)
    if wm_font_size is None:
        wm_font_size = int(size[1] * 2.0)  # 텍스트 높이에 비례해서 워터마크 글자 크기 설정

    wm_pattern = generate_watermark_pattern(
        wm_text, size=size, font_path=font_path, font_size=wm_font_size,
        angle_deg=angle_deg, gray_value=gray_value,
    )

    # 원본 글자의 "글자 영역(text_mask)"을 먼저 계산 (합성 전 원본 기준)
    text_arr = np.array(text_img.convert("L"))
    text_mask = get_text_mask(text_arr)

    # 1) 흰 배경 캔버스에 워터마크를 먼저 붙인다
    composite = Image.new("RGB", size, color=(255, 255, 255))
    # x_shift만큼 밀어서 붙이되, 캔버스를 벗어나지 않게 crop
    wm_shifted = Image.new("RGB", size, color=(255, 255, 255))
    wm_shifted.paste(wm_pattern, box=(x_shift, 0))
    composite.paste(wm_shifted, box=(0, 0))

    # 2) 워터마크가 차지하는 영역(wm_mask) 계산 - 배경(255)이 아닌 모든 픽셀
    wm_mask = np.array(composite.convert("L")) != 255

    # 3) 그 위에 원본 글자를 text_mask를 이용해 덮어씌운다
    #    (text_mask가 True인 곳만 원본 글자색으로 바뀌고, 나머지는 워터마크가 유지됨)
    text_mask_img = Image.fromarray((text_mask * 255).astype(np.uint8))
    composite.paste(text_img, mask=text_mask_img)

    # 글자가 덮어쓴 영역은 더 이상 워터마크가 아니므로 wm_mask에서 제외
    wm_mask = wm_mask & (~text_mask)

    return {
        "composite": composite,
        "wm_mask": wm_mask,
        "text_mask": text_mask,
        "wm_pattern": wm_pattern,
    }
