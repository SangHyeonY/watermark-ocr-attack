# -*- coding: utf-8 -*-
"""
토익 5문항 베이스라인 측정 (1/2): OCR 문자 인식 결과 수집.

Step B의 measure_ocr_baseline.py와 로직은 동일하되, toeic_questions.json과
토익 문항 이미지를 대상으로 한다. 워터마크/공격을 적용하지 않은 원본 이미지를
직접 렌더링해서 측정한다 (Step B의 question_images 폴더처럼 별도 렌더링
단계 없이, 이 스크립트 안에서 바로 렌더링 -> 측정까지 수행).

실행 방법:
    conda activate fawa
    python pytorch_port/step_c/measure_ocr_baseline_toeic.py
"""
import json
import os
import sys

import easyocr
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from text_metrics import compute_cer  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "step_b"))
from render_questions import render_question_image, CHOICE_LABELS  # noqa: E402

QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "toeic_questions.json")
RESULT_PATH = os.path.join(os.path.dirname(__file__), "outputs_toeic", "ocr_baseline_result.json")


def load_questions():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_reference_text(question_obj):
    parts = [f"{question_obj['id']}.", question_obj["question"]]
    for label, choice in zip(CHOICE_LABELS, question_obj["choices"]):
        parts.append(label)
        parts.append(choice)
    return "".join(parts)


def main():
    questions = load_questions()
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)

    print("[info] EasyOCR Reader 로딩 중 (ko+en)...")
    reader = easyocr.Reader(["ko", "en"], gpu=False)

    results = []
    total_cer = 0.0

    for q in questions:
        img = render_question_image(q)
        img_arr = np.array(img.convert("RGB"))

        ocr_result = reader.readtext(img_arr, detail=0, paragraph=False)
        recognized_text = "".join(ocr_result)

        reference_text = build_reference_text(q)
        recognized_clean = recognized_text.replace(" ", "")
        reference_clean = reference_text.replace(" ", "")

        cer = compute_cer(recognized_clean, reference_clean)
        total_cer += cer

        results.append({
            "id": q["id"],
            "subject": q["subject"],
            "reference_text": reference_text,
            "recognized_text": recognized_text,
            "cer": round(cer, 4),
        })

        print(f"  q{q['id']} ({q['subject']}): CER={cer:.4f}")
        print(f"    정답: {reference_text}")
        print(f"    OCR : {recognized_text}")

    avg_cer = total_cer / len(questions)
    summary = {
        "num_questions": len(questions),
        "average_cer": round(avg_cer, 4),
        "results": results,
    }

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[완료] 평균 CER: {avg_cer:.4f}")
    print(f"[결과 저장] {RESULT_PATH}")


if __name__ == "__main__":
    main()
