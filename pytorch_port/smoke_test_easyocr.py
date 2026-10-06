# -*- coding: utf-8 -*-
"""
스모크 테스트: 한글 단어 이미지를 생성하고, EasyOCR(PyTorch 기반)로
정상적으로 인식되는지 확인한다.

목적:
- PyTorch + EasyOCR 환경이 제대로 설치되었는지 확인
- 한글 OCR이 실제로 동작하는지 1차 검증
- 이후 FAWA 공격(워터마크 기반 섭동) 코드를 붙이기 전에,
  "공격 대상이 될 OCR 파이프라인"이 정상 동작하는지 베이스라인으로 확보

참고 (중요):
- 원래는 trdg(TextRecognitionDataGenerator, 원본 FAWA 코드가 사용하던 라이브러리)의
  GeneratorFromStrings로 이미지를 생성하려 했으나, trdg는 한글 전용 폰트 폴더가
  없고 내부 왜곡/배경 합성 파이프라인이 한글 렌더링을 깨뜨리는 문제가 있었다
  (글자 일부가 잘리거나 받침이 누락됨 -> OCR이 "학교"를 "한i"로 잘못 인식하는 등).
  이를 PIL로 직접 텍스트를 렌더링하는 방식으로 대체하여 문제를 해결했다
  (아래 generate_word_images 참고). 즉 trdg 의존을 제거했다.

실행 방법:
    conda activate fawa
    python pytorch_port/smoke_test_easyocr.py
"""
import os
import sys

import numpy as np
import torch
import easyocr
from PIL import Image, ImageDraw, ImageFont

# 테스트에 쓸 한국어 단어 목록 (받아쓰기 쉬운 짧은 단어들)
TEST_WORDS = ["시험", "문제", "정답", "학교", "한국어"]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "smoke_test")

# Windows에 기본 설치된 한글 폰트(맑은 고딕 등)를 WSL 마운트 경로로 직접 지정한다.
KOREAN_FONT_CANDIDATES = [
    "/mnt/c/Windows/Fonts/malgun.ttf",
    "/mnt/c/Windows/Fonts/NanumGothic.ttf",
]

FONT_SIZE = 60
MARGIN = 20


def _resolve_korean_font():
    for path in KOREAN_FONT_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "사용 가능한 한글 폰트를 찾지 못했습니다. "
        f"다음 경로 중 하나에 폰트가 있어야 합니다: {KOREAN_FONT_CANDIDATES}"
    )


def generate_word_images(words):
    """PIL로 흰 배경에 검은 글자 한글 단어 이미지를 직접 렌더링한다.

    (trdg 대신 PIL을 직접 쓰는 이유는 파일 상단 docstring 참고)
    """
    font_path = _resolve_korean_font()
    font = ImageFont.truetype(font_path, FONT_SIZE)

    images = []
    for word in words:
        bbox = font.getbbox(word)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        img_w, img_h = text_w + 2 * MARGIN, text_h + 2 * MARGIN
        img = Image.new("RGB", (img_w, img_h), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((MARGIN - bbox[0], MARGIN - bbox[1]), word, fill=(0, 0, 0), font=font)
        images.append((word, img))
    return images


def main():
    print(f"[info] torch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[info] 한글 단어 이미지 생성 중 (PIL 직접 렌더링)...")
    word_images = generate_word_images(TEST_WORDS)
    for word, img in word_images:
        save_path = os.path.join(OUTPUT_DIR, f"{word}.png")
        img.save(save_path)
    print(f"[info] {len(word_images)}개 이미지 생성 완료 -> {OUTPUT_DIR}")

    print("[info] EasyOCR Reader 로딩 중 (ko+en, CPU)...")
    reader = easyocr.Reader(["ko", "en"], gpu=False)

    print("[info] OCR 인식 시작...")
    correct = 0
    for word, img in word_images:
        arr = np.array(img.convert("RGB"))
        result = reader.readtext(arr, detail=1)
        predicted_texts = [text for (_, text, _) in result]
        predicted = "".join(predicted_texts)
        is_match = predicted.strip() == word
        correct += int(is_match)
        status = "OK" if is_match else "MISMATCH"
        print(f"  [{status}] 정답: '{word}' / OCR 결과: '{predicted}'")

    total = len(word_images)
    print(f"\n[결과] 정확히 일치한 단어: {correct}/{total}")

    if correct == 0:
        print("[경고] 하나도 맞지 않았습니다. 폰트/전처리 문제일 수 있으니 생성된 이미지를 직접 확인하세요.")
        sys.exit(1)
    else:
        print("[성공] EasyOCR + PyTorch 환경이 정상 동작합니다.")


if __name__ == "__main__":
    main()
