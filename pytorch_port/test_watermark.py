# -*- coding: utf-8 -*-
"""
watermark.py 검증 스크립트.

확인할 것:
1. 워터마크 합성 후에도 사람이 읽는 글자(text_mask 영역)는 원본과 동일하게 유지되는가.
2. wm_mask / text_mask가 서로 겹치지 않고(배타적), 합쳐서 전체 이미지 영역을 설명하는가.
3. 워터마크를 합성한 이미지를 EasyOCR에 넣었을 때도 원본 글자를 올바르게 인식하는가
   (아직 공격 섭동을 가하지 않은 "워터마크 배경 추가 전/후" 비교이므로, 이 단계에서는
   인식 결과가 원본과 같아야 정상이다 - 아직 적대적 섭동을 넣지 않았기 때문).

실행 방법:
    conda activate fawa
    python pytorch_port/test_watermark.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from watermark import render_text_image, composite_text_with_watermark

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "watermark_test")

TEST_WORDS = ["시험", "문제", "정답", "학교", "한국어"]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    import easyocr
    print("[info] EasyOCR Reader 로딩 중 (ko+en, CPU)...")
    reader = easyocr.Reader(["ko", "en"], gpu=False)

    all_ok = True
    for word in TEST_WORDS:
        text_img = render_text_image(word, font_size=60, margin=20)
        result = composite_text_with_watermark(
            text_img, wm_text="PROTECTED", angle_deg=10, gray_value=220, x_shift=10,
        )
        composite = result["composite"]
        wm_mask = result["wm_mask"]
        text_mask = result["text_mask"]

        # 저장 (시각 확인용)
        text_img.save(os.path.join(OUTPUT_DIR, f"{word}_original.png"))
        composite.save(os.path.join(OUTPUT_DIR, f"{word}_composite.png"))
        result["wm_pattern"].save(os.path.join(OUTPUT_DIR, f"{word}_wm_pattern.png"))

        # 마스크 배타성 확인
        overlap = np.sum(wm_mask & text_mask)
        wm_ratio = wm_mask.mean()
        text_ratio = text_mask.mean()

        # OCR 인식 확인 (아직 적대적 섭동 없음 -> 원본과 동일하게 읽혀야 정상)
        arr = np.array(composite.convert("RGB"))
        ocr_result = reader.readtext(arr, detail=1)
        predicted = "".join(t for (_, t, _) in ocr_result)
        ocr_match = predicted.strip() == word

        status = "OK" if (overlap == 0 and ocr_match) else "CHECK"
        print(
            f"[{status}] '{word}': wm_mask 비율={wm_ratio:.1%}, "
            f"text_mask 비율={text_ratio:.1%}, 마스크 겹침 픽셀={overlap}, "
            f"OCR 결과='{predicted}' (일치={ocr_match})"
        )
        if overlap != 0 or not ocr_match:
            all_ok = False

    print(f"\n[결과 요약] 저장 위치: {OUTPUT_DIR}")
    if all_ok:
        print("[성공] 워터마크 합성 로직이 정상 동작합니다 (마스크 배타적, OCR 인식 유지).")
    else:
        print("[경고] 일부 항목에서 문제가 발견되었습니다. 위 로그와 저장된 이미지를 확인하세요.")
        sys.exit(1)


if __name__ == "__main__":
    main()
