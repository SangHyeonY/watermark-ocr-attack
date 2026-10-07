# -*- coding: utf-8 -*-
"""
Step B 베이스라인 측정 (1/2): OCR 문자 인식 결과 수집.

render_questions.py로 만든 20문항 이미지를 EasyOCR에 넣어서, 보호 패턴을
적용하지 않은 "원본" 상태의 문자 인식 정확도를 측정한다. 이후 Step C에서
워터마크 적용 후 결과와 비교하기 위한 기준점(baseline)이다.

측정 방식:
- 각 문항 이미지 전체(문제+선택지)를 EasyOCR에 통째로 넣어 인식시킨다.
- EasyOCR이 인식한 전체 텍스트와, 원본 텍스트(questions.json에서 재구성)를
  비교해 CER(Character Error Rate, 문자 단위 오류율)을 계산한다.
  CER이 낮을수록 원본과 가깝게 인식했다는 뜻이다.

실행 방법:
    conda activate fawa
    python pytorch_port/step_b/measure_ocr_baseline.py
"""
import json
import os
import sys

import easyocr
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from text_metrics import levenshtein_distance, compute_cer  # noqa: E402

QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "questions.json")
IMAGE_DIR = os.path.join(os.path.dirname(__file__), "outputs", "question_images")
RESULT_PATH = os.path.join(os.path.dirname(__file__), "outputs", "ocr_baseline_result.json")

CHOICE_LABELS = ["A)", "B)", "C)", "D)"]
# render_questions.py와 동일한 라벨 형식으로 맞춤 (WORK_LOG_StepC.md 참고)


def load_questions():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_reference_text(question_obj):
    """questions.json의 문항 데이터로부터, 이미지에 그려진 것과 동일한 순서의
    기준 텍스트(정답 비교용)를 재구성한다. 공백 없이 이어붙여 CER 계산을
    단순화한다 (EasyOCR 결과도 같은 방식으로 이어붙여 비교)."""
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
        image_path = os.path.join(IMAGE_DIR, f"q{q['id']:02d}.png")
        img = Image.open(image_path).convert("RGB")
        img_arr = np.array(img)

        ocr_result = reader.readtext(img_arr, detail=0, paragraph=False)
        recognized_text = "".join(ocr_result)

        reference_text = build_reference_text(q)
        # 공백은 OCR 결과에서 줄 구분에 따라 들쭉날쭉할 수 있어 비교에서 제외
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

        print(f"  q{q['id']:02d} ({q['subject']}): CER={cer:.4f}")
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
