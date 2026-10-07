# -*- coding: utf-8 -*-
"""
Step C: 20문항에 워터마크 보호 패턴 + 적대적 공격을 적용한다.

배경 (쉬운 설명):
- 1차 설계: 문제를 "문제 1줄 + 선택지 4줄"로 미리 쪼개서 각 줄을 따로 렌더링하고
  공격했다. 그런데 이렇게 하면 글자 주변 여백(margin) 비율이 Step B의 원본과
  달라져서, 공격을 걸기도 전에 OCR이 이미 많이 틀리는 문제가 발견되었다
  (예: "A) 이성계" -> "세 이성계"). 반면 Step B에서는 같은 글자를 EasyOCR의
  readtext() 고수준 API로 인식시키면 정확히 읽혔다.
- 원인 확인: readtext()는 내부적으로 먼저 CRAFT 검출 모델로 "글자가 정확히
  어디 있는지" 찾아내고, 그 영역만 여백 없이 딱 잘라서 인식 모델에 넣는다.
  우리가 직접 만든 저수준 경로는 이 검출 단계를 생략하고 "직접 렌더링한, 여백이
  들어간" 이미지를 그대로 넣었던 것이 품질 차이의 원인이었다.
- 해결한 설계 (2차, 현재 버전): Step B와 동일하게 문제 전체(5줄)를 하나의 큰
  이미지로 렌더링하고, 거기에 워터마크를 한 번에 합성한다. 그 다음 EasyOCR의
  검출 모델(reader.detect)로 각 줄의 정확한 바운딩 박스를 찾아서, 그 박스
  영역만 정밀하게 잘라내 공격을 수행한다. 공격이 끝난 조각을 원래 위치에
  다시 붙여넣어 최종 이미지를 완성한다. 이렇게 하면 Step B가 잘 동작했던
  것과 동일한 "검출 후 인식" 경로를 그대로 활용하면서, 검출된 영역에만
  공격을 가할 수 있다.

강도(eps) 변수 실험:
- 작업정리 문서가 요구하는 "강도·크기·배치·적용 영역" 변수 중, 1차로는
  "강도"(eps, 공격 허용 변화폭)만 3단계(0.3 / 0.45 / 0.6)로 비교한다.

실행 방법:
    conda activate fawa
    python pytorch_port/step_c/attack_questions.py
"""
import json
import math
import os
import sys

import easyocr
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from watermark import composite_text_with_watermark  # noqa: E402
from ocr_model import OcrAttackModel, tensor_to_pil_image, IMG_HEIGHT  # noqa: E402
from attack import attack_untargeted  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "step_b"))
from render_questions import load_questions, render_question_image, CHOICE_LABELS  # noqa: E402

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

EPS_CONDITIONS = {
    "low": 0.3,
    "mid": 0.45,  # Step A에서 채택한 기본값
    "high": 0.6,
}

ATTACK_MAX_ITER = 150


def detect_line_boxes(reader, pil_image):
    """EasyOCR 검출 모델로 이미지 안의 각 줄(텍스트박스)의 바운딩 박스를 찾는다.

    Returns:
        list of (x_min, y_min, x_max, y_max) - 위에서 아래 순서로 정렬됨.
    """
    arr = np.array(pil_image.convert("RGB"))
    horizontal_list, free_list = reader.detect(arr)
    boxes = []
    for box in horizontal_list[0]:
        x_min, x_max, y_min, y_max = box
        boxes.append((int(x_min), int(y_min), int(x_max), int(y_max)))
    # 위에서 아래 순서로 정렬 (문제 -> 선택지 A,B,C,D 순서가 되도록)
    boxes.sort(key=lambda b: b[1])
    return boxes


def attack_region(wrapper, region_img, wm_mask_region, ground_truth_text, eps, eps_iter, verbose=False):
    """crop된 한 영역(이미 워터마크가 합성된 상태)에 대해 공격을 수행한다.

    Args:
        region_img: PIL.Image (RGB) - 검출 박스로 crop된 영역 (워터마크 합성 후).
        wm_mask_region: np.ndarray(bool) - 같은 영역에 대응하는 wm_mask 조각.
        ground_truth_text: 이 영역의 실제 정답 텍스트. CER 계산 기준으로 쓰인다
            (attack.py의 3차 성공 판정 기준 - WORK_LOG_StepC.md 참고).
        eps, eps_iter: attack_untargeted에 전달할 공격 강도 파라미터.

    Returns:
        dict: {"image": PIL.Image, "success": bool, "ocr_before": str, "ocr_after": str,
               "cer_before": float, "cer_after": float}
    """
    w, h = region_img.size
    needed_width = math.ceil(IMG_HEIGHT * w / h) + 10

    original_tensor = wrapper.image_to_tensor(region_img, max_width=needed_width)
    wm_mask_tensor = wrapper.mask_to_tensor(wm_mask_region, region_img.size, max_width=needed_width)

    baseline_pred = wrapper.predict_text(original_tensor)

    result = attack_untargeted(
        ocr_wrapper=wrapper,
        original_tensor=original_tensor,
        wm_mask_tensor=wm_mask_tensor,
        original_text=baseline_pred if baseline_pred else "x",
        ground_truth_text=ground_truth_text,
        eps=eps,
        eps_iter=eps_iter,
        max_iter=ATTACK_MAX_ITER,
        verbose=verbose,
    )

    ocr_before = result["baseline_prediction"]
    ocr_after = result["final_prediction"]

    attacked_small = tensor_to_pil_image(result["adv_tensor"])
    scale = h / float(IMG_HEIGHT)
    valid_w = min(needed_width, round(w / scale))
    attacked_cropped = attacked_small.crop((0, 0, valid_w, IMG_HEIGHT))
    attacked_resized = attacked_cropped.resize((w, h), Image.BICUBIC).convert("RGB")

    return {
        "image": attacked_resized,
        "success": result["success"],
        "ocr_before": ocr_before,
        "ocr_after": ocr_after,
        "cer_before": result["baseline_cer"],
        "cer_after": result["final_cer"],
    }


def build_ground_truth_texts(question_obj):
    """문항 하나로부터, 검출될 것으로 예상되는 각 줄의 정답 텍스트 목록을 만든다.
    render_question_image()가 "문제, 선택지A, 선택지B, 선택지C, 선택지D" 순서로
    그리므로, 검출 박스도 위->아래 순서로 정렬하면 이 순서와 대응된다.

    주의: CRAFT 검출기가 "문항 번호"와 "질문 본문"을 별도 박스로 나누는 경우가
    있어(관찰됨), 박스 개수가 5개보다 많을 수 있다. 이 경우 박스 개수와 정답
    개수가 안 맞을 수 있으므로, 개수가 다르면 그 문항은 "문항 번호 분리" 등의
    예외로 보고 정답 매칭을 생략한다 (아래 attack_question_image에서 처리).
    """
    question_line = f"{question_obj['id']}. {question_obj['question']}"
    choice_lines = [
        f"{CHOICE_LABELS[i]} {choice}"
        for i, choice in enumerate(question_obj["choices"])
    ]
    return [question_line] + choice_lines


def attack_question_image(wrapper, reader, question_obj, eps, eps_iter, verbose=False):
    """문항 하나(전체 문제 이미지)에 대해 검출 -> 영역별 공격 -> 재조합을 수행한다.

    Returns:
        dict: {"image": PIL.Image, "line_results": list of dict}
    """
    # 1) Step B와 동일한 방식으로 문제 전체(5줄) 이미지 렌더링
    question_img = render_question_image(question_obj)

    # 2) 전체 이미지에 워터마크 한 번 합성
    comp = composite_text_with_watermark(
        question_img, wm_text="PROTECTED", angle_deg=10, gray_value=220, x_shift=10,
    )
    composite_img = comp["composite"]
    wm_mask_full = comp["wm_mask"]  # numpy bool array, shape (H, W)

    # 3) 검출 모델로 각 줄의 바운딩 박스 찾기 (합성 이미지 기준)
    boxes = detect_line_boxes(reader, composite_img)

    # 4) 박스 순서(위->아래)와 정답 텍스트 순서(문제, A, B, C, D)를 매칭.
    #    보통 5개 박스(문제, A, B, C, D)가 검출되지만, CRAFT 검출기가 "문항 번호"
    #    (예: "1.")와 "질문 본문"을 별도 박스 2개로 나누는 경우가 20문항 중 4개
    #    (20%) 꼴로 있었다 (WORK_LOG_StepC.md 참고). 이 경우 박스가 6개가 되는데,
    #    첫 번째 박스(번호만)는 "N."으로, 두 번째 박스(본문)는 질문 텍스트만으로
    #    따로 매칭한다 (번호 박스에 전체 문제 텍스트를 매칭하면 CER이 비정상적으로
    #    높게 나오는 왜곡이 있었음 - 숫자 하나짜리 텍스트와 긴 정답을 비교하게 되므로).
    expected_texts = build_ground_truth_texts(question_obj)
    if len(boxes) == len(expected_texts):
        ground_truths = expected_texts
    elif len(boxes) == len(expected_texts) + 1:
        number_only = f"{question_obj['id']}."
        question_only = question_obj["question"]
        # 맨 위 2개 박스는 "번호"와 "본문"인데, y좌표(위아래 순서)로는 어느 쪽이
        # 번호인지 보장되지 않는다 (관찰 결과 번호 박스가 본문 박스보다 y_min이
        # 더 큰(더 아래) 경우가 있었음 - 아마 베이스라인 정렬 차이). 대신 번호는
        # 항상 본문보다 왼쪽(x_min이 더 작음)에 있으므로 x좌표로 구분한다.
        first_box, second_box = boxes[0], boxes[1]
        if first_box[0] <= second_box[0]:  # x_min 비교
            top_two_truths = [number_only, question_only]
        else:
            top_two_truths = [question_only, number_only]
        ground_truths = top_two_truths + expected_texts[1:]
        if verbose:
            print(
                f"  [정보] q{question_obj['id']}: 검출된 박스 수({len(boxes)})가 예상보다 "
                f"1개 많음 (문항 번호가 분리된 것으로 추정). 번호/본문을 각각 매칭."
            )
    else:
        ground_truths = [None] * len(boxes)
        if verbose:
            print(
                f"  [주의] q{question_obj['id']}: 검출된 박스 수({len(boxes)})가 "
                f"예상 줄 수({len(expected_texts)})와 크게 다름. ground_truth 매칭 생략."
            )

    # 5) 각 박스 영역을 crop해서 개별적으로 공격
    final_img = composite_img.copy()
    line_results = []
    for (x_min, y_min, x_max, y_max), gt in zip(boxes, ground_truths):
        region = composite_img.crop((x_min, y_min, x_max, y_max))
        wm_mask_region = wm_mask_full[y_min:y_max, x_min:x_max]

        res = attack_region(wrapper, region, wm_mask_region, gt, eps, eps_iter, verbose=verbose)
        final_img.paste(res["image"], box=(x_min, y_min))

        line_results.append({
            "box": [x_min, y_min, x_max, y_max],
            "ground_truth": gt,
            "ocr_before": res["ocr_before"],
            "ocr_after": res["ocr_after"],
            "cer_before": res["cer_before"],
            "cer_after": res["cer_after"],
            "success": res["success"],
        })

    return {"image": final_img, "line_results": line_results}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    questions = load_questions()

    print("[info] EasyOCR Reader 로딩 중 (quantize=False, 공격용 설정)...")
    reader = easyocr.Reader(["ko", "en"], gpu=False, quantize=False)
    wrapper = OcrAttackModel(reader)

    for condition_name, eps in EPS_CONDITIONS.items():
        eps_iter = eps * 0.1
        condition_dir = os.path.join(OUTPUT_DIR, f"question_images_{condition_name}")
        os.makedirs(condition_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"[조건: {condition_name}] eps={eps}, eps_iter={eps_iter}")
        print(f"{'='*60}")

        condition_log = []

        for q in questions:
            result = attack_question_image(wrapper, reader, q, eps, eps_iter)
            save_path = os.path.join(condition_dir, f"q{q['id']:02d}.png")
            result["image"].save(save_path)

            line_results = result["line_results"]
            success_count = sum(1 for r in line_results if r["success"])
            matched_count = sum(1 for r in line_results if r["ground_truth"] is not None)
            avg_cer_before = (
                sum(r["cer_before"] for r in line_results if r["ground_truth"] is not None) / matched_count
                if matched_count else None
            )
            avg_cer_after = (
                sum(r["cer_after"] for r in line_results if r["ground_truth"] is not None) / matched_count
                if matched_count else None
            )
            print(
                f"  q{q['id']:02d} ({q['subject']}): "
                f"{len(line_results)}개 영역 검출({matched_count}개 정답 매칭됨), "
                f"{success_count}개 공격 성공, "
                f"평균 CER {avg_cer_before:.3f}->{avg_cer_after:.3f}" if matched_count else
                f"  q{q['id']:02d} ({q['subject']}): 정답 매칭 실패"
            )
            for r in line_results:
                print(
                    f"      box={r['box']} gt={r['ground_truth']!r} "
                    f"before={r['ocr_before']!r}(cer={r['cer_before']:.2f}) "
                    f"after={r['ocr_after']!r}(cer={r['cer_after']:.2f}) success={r['success']}"
                )

            condition_log.append({
                "id": q["id"],
                "subject": q["subject"],
                "num_regions": len(line_results),
                "num_matched": matched_count,
                "regions_attacked": success_count,
                "avg_cer_before": avg_cer_before,
                "avg_cer_after": avg_cer_after,
                "line_details": line_results,
            })

        log_path = os.path.join(condition_dir, "attack_log.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(condition_log, f, ensure_ascii=False, indent=2)
        print(f"[조건: {condition_name}] 완료, 로그 저장 -> {log_path}")

    print("\n[전체 완료] 모든 eps 조건에 대한 공격된 문제 이미지 생성 완료.")


if __name__ == "__main__":
    main()
