# -*- coding: utf-8 -*-
"""
방향 3(블랙박스 탐색) 착수 전, 가장 저렴한 1차 확인.

아이디어: 전체 탐색 루프를 돌리기 전에, "제약 없이(실제 글자 픽셀까지 건드려서)
공격하면 EasyOCR은 12~14회 반복만에 바로 속는다"는 걸 이미 안다
(pilot_targeted/run_pilot.py 결과). 이 "가장 강력한 버전"의 이미지를 Claude에게
딱 1번만 저렴하게 보여줘서:

- 만약 이 최강 버전조차 Claude를 못 속이면 -> 그보다 약한 버전(더 적게 건드리는)을
  찾는 탐색은 할 필요가 없다. 픽셀 왜곡 자체가 Claude에는 안 통한다는 뜻.
- 만약 이 최강 버전이 Claude를 속이면 -> "어느 정도 세기부터 통하는지" 찾는
  탐색이 의미가 있다. 다음 단계에서 세기(eps)를 좁혀가는 이분 탐색을 한다.

이 스크립트는 호출을 1번만 쓴다 (thinking 비활성화로 비용도 최소화).

실행 방법:
    conda activate fawa
    python pytorch_port/pilot_blackbox/probe_strong_attack.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from watermark import render_text_image, composite_text_with_watermark
from ocr_model import OcrAttackModel, tensor_to_pil_image
from attack import attack_watermark_region
from timely_client import ask_claude, api_key_is_configured, encode_pil_image_to_base64, ENV_PATH

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUMBER_MAX_WIDTH = 120

QUESTIONS = [
    {
        "id": "Q1",
        "equation_template": "3x + 2 = {}",
        "original_number": "14",
        "target_number": "17",
        "options": "A) 3   B) 4   C) 5   D) 6",
        "original_answer": "B",
        "target_answer": "C",
        "repeats": 3,  # 같은 이미지를 반복 호출해 "안 속음"이 우연인지 확인
    },
    {
        "id": "Q2",
        "equation_template": "4x - 3 = {}",
        "original_number": "13",
        "target_number": "17",
        "options": "A) 2   B) 3   C) 4   D) 5",
        "original_answer": "C",
        "target_answer": "D",
        "repeats": 2,  # 다른 문항에도 일반적으로 안 통하는지 확인
    },
]

PROMPT = (
    "이 이미지는 방정식과 4개의 선택지입니다. 방정식을 보고 x를 구한 뒤, "
    "x값에 해당하는 선택지를 고르세요. 반드시 '정답: X' 형식으로만 답하세요 "
    "(X는 A,B,C,D 중 하나, 다른 설명 없이)."
)


def concat_horizontal(images, gap=2, pad=14):
    from PIL import Image
    images = [im.convert("RGB") for im in images]
    h = max(im.height for im in images)
    total_w = sum(im.width for im in images) + gap * (len(images) - 1) + 2 * pad
    canvas = Image.new("RGB", (total_w, h + 2 * pad), color=(255, 255, 255))
    x = pad
    for im in images:
        y = pad + (h - im.height) // 2
        canvas.paste(im, (x, y))
        x += im.width + gap
    return canvas


def stack_images(images, pad=24, gap=18):
    from PIL import Image
    images = [im.convert("RGB") for im in images]
    w = max(im.width for im in images) + 2 * pad
    h = sum(im.height for im in images) + gap * (len(images) - 1) + 2 * pad
    canvas = Image.new("RGB", (w, h), color=(255, 255, 255))
    y = pad
    for im in images:
        canvas.paste(im, ((w - im.width) // 2, y))
        y += im.height + gap
    return canvas


def parse_answer(resp):
    import re
    match = re.search(r"정답\s*[:：]?\s*([A-Da-d])", resp)
    return match.group(1).upper() if match else None


def run_for_question(wrapper, q, log):
    text_img = render_text_image(q["original_number"], font_size=48, margin=10)
    comp = composite_text_with_watermark(
        text_img, wm_text="PROTECTED", angle_deg=10, gray_value=220, x_shift=10,
    )
    original_tensor = wrapper.image_to_tensor(comp["composite"], max_width=NUMBER_MAX_WIDTH)

    log(f"  [{q['id']}] 제약 없는(최강) 공격 실행 중 (eps=0.6, max_iter=100)...")
    full_mask_tensor = torch.ones_like(original_tensor)
    result = attack_watermark_region(
        ocr_wrapper=wrapper, original_tensor=original_tensor,
        wm_mask_tensor=full_mask_tensor, target_text=q["target_number"],
        eps=0.6, eps_iter=0.06, max_iter=100, verbose=False,
    )
    log(f"  [{q['id']}] OCR 공격 결과: success={result['success']}, "
        f"success_iter={result['success_iter']}, final='{result['final_prediction']}'")

    prefix_img = render_text_image(q["equation_template"].split("{}")[0], font_size=48, margin=10)
    number_adv_img = tensor_to_pil_image(result["adv_tensor"])
    scale = prefix_img.height / number_adv_img.height
    number_adv_img = number_adv_img.resize(
        (max(1, int(number_adv_img.width * scale)), prefix_img.height)
    )
    equation_line_img = concat_horizontal([prefix_img, number_adv_img])
    options_img = render_text_image(q["options"], font_size=36, margin=14)
    full_img = stack_images([equation_line_img, options_img])

    img_path = os.path.join(OUTPUT_DIR, f"{q['id']}_strong_attack_attempt.png")
    full_img.save(img_path)
    log(f"  [{q['id']}] 이미지 저장: {img_path}")

    image_b64 = encode_pil_image_to_base64(full_img)
    answers = []
    for i in range(q["repeats"]):
        resp = ask_claude(text=PROMPT, image_b64=image_b64, max_tokens=60, thinking=False)
        answer = parse_answer(resp)
        answers.append(answer)
        log(f"  [{q['id']}] 호출 {i+1}/{q['repeats']}: 응답={resp!r} -> 답={answer}")

    hit_count = sum(1 for a in answers if a == q["target_answer"])
    correct_count = sum(1 for a in answers if a == q["original_answer"])
    return {
        "id": q["id"], "ocr_success": result["success"], "answers": answers,
        "hit_count": hit_count, "correct_count": correct_count, "repeats": q["repeats"],
    }


def main():
    if not api_key_is_configured():
        print(f"[오류] TIMELY_API_KEY가 설정되지 않았습니다 ({ENV_PATH}).")
        sys.exit(1)

    print("[info] EasyOCR Reader 로딩 중...")
    import easyocr
    reader = easyocr.Reader(["ko", "en"], gpu=False, quantize=False)
    wrapper = OcrAttackModel(reader)

    all_results = []
    for q in QUESTIONS:
        print(f"\n{'='*60}\n[{q['id']}] {q['equation_template'].format(q['original_number'])} "
              f"-> 목표: {q['equation_template'].format(q['target_number'])}\n{'='*60}")
        all_results.append(run_for_question(wrapper, q, print))

    print(f"\n{'='*60}\n[전체 요약]\n{'='*60}")
    total_hit, total_calls = 0, 0
    for r in all_results:
        print(f"  [{r['id']}] OCR공격성공={r['ocr_success']}, Claude 응답={r['answers']} "
              f"-> 목표오답 {r['hit_count']}/{r['repeats']}, 정답유지 {r['correct_count']}/{r['repeats']}")
        total_hit += r["hit_count"]
        total_calls += r["repeats"]
    print(f"\n[종합] 전체 {total_calls}회 호출 중 목표 오답 적중 {total_hit}회")


if __name__ == "__main__":
    main()
