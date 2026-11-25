# -*- coding: utf-8 -*-
"""
초소형 파일럿: "FAWA targeted 공격(EasyOCR 대상)이 Claude 같은 멀티모달 AI에게도
전이(transfer)되어, 지정한 오답을 고르게 만드는가?"를 1~2문항으로 빠르게 확인한다.

배경 (WORK_LOG_StepC/StepC_TOEIC):
- untargeted 공격으로 OCR CER을 0.6까지 올려도 Claude 정답률은 100% 그대로였다.
- 이번에는 목표를 "OCR을 틀리게"에서 "AI가 특정 숫자로 잘못 읽고, 그 결과
  지정한 오답을 고르게"로 바꿔서, FAWA의 기존 targeted 공격
  (attack.py의 attack_watermark_region)이 실제로 쓸만한지부터 확인한다.

이 스크립트가 하는 일 (두 단계):

[1단계: OCR 쪽 표적 공격이 애초에 가능한지 진단]
  - 숫자 하나(예: "14")만 떼어내 워터마크를 합성하고, 워터마크 영역(wm_mask)에만
    섭동을 가해 "17"로 읽히도록 attack_watermark_region()을 돌린다.
  - 비교용으로, 제약을 없앤(실제 글자 픽셀까지 건드리는) 버전도 똑같이 돌려서
    "방법 자체는 문제없는지" 확인한다. (진단용, 사람 가독성은 포기한 상태)

[2단계: "목표 오답 설계" 자체가 말이 되는지 Claude로 직접 확인]
  - 워터마크 기술과 분리해서, "숫자를 실제로 다른 값으로 바꿔 쓰면 Claude가
    정말로 의도한 오답을 고르는가?"를 직접 테스트한다 (사니티 체크/상한선).
  - 베이스라인(원본), 목표 숫자로 실제 치환한 이미지, (1단계 결과에 따라)
    워터마크로 유도를 시도한 이미지까지 Claude에게 보여주고 답을 비교한다.

실행 방법:
    conda activate fawa  (또는 ~/miniconda3/bin/activate fawa)
    python pytorch_port/pilot_targeted/run_pilot.py
"""
import json
import os
import re
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from watermark import render_text_image, composite_text_with_watermark
from ocr_model import OcrAttackModel, tensor_to_pil_image
from attack import attack_watermark_region
from timely_client import (
    ask_claude as timely_ask_claude,
    api_key_is_configured,
    encode_pil_image_to_base64,
    ENV_PATH,
)

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPS_CANDIDATES = [0.3, 0.6, 1.0]
MAX_ITER_MASKED = 200       # 워터마크 영역만 공격 (실제 제약 조건)
MAX_ITER_UNMASKED = 100     # 진단용: 제약 없이 전체 이미지 공격

NUMBER_MAX_WIDTH = 120  # "14" 같은 두 글자 숫자용 텐서 폭

# ---------------------------------------------------------------------------
# 문항 설계 (숫자 검산 포함)
# ---------------------------------------------------------------------------
QUESTIONS = [
    {
        "id": "Q1",
        "equation_template": "3x + 2 = {}",
        "original_number": "14",
        "target_number": "17",
        "options": "A) 3   B) 4   C) 5   D) 6",
        "original_answer": "B",  # 3*4+2=14
        "target_answer": "C",    # 3*5+2=17
    },
    {
        "id": "Q2",
        "equation_template": "4x - 3 = {}",
        "original_number": "13",
        "target_number": "17",
        "options": "A) 2   B) 3   C) 4   D) 5",
        "original_answer": "C",  # 4*4-3=13
        "target_answer": "D",    # 4*5-3=17
    },
]

PROMPT_TEMPLATE = (
    "이 이미지는 방정식과 4개의 선택지입니다. 방정식을 보고 x를 구한 뒤, "
    "x값에 해당하는 선택지를 고르세요.\n"
    "반드시 아래 형식으로만 답하세요 (다른 설명 없이):\n"
    "정답: X\n"
    "(X는 A, B, C, D 중 하나)"
)
LETTERS = ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# 1단계: OCR 표적 공격 진단
# ---------------------------------------------------------------------------
def diagnose_ocr_targeted_attack(wrapper, original_number, target_number, log):
    """숫자 하나에 대해 (a) 워터마크 영역만 공격(실제 제약) (b) 제약 없이 전체
    공격(진단용)을 둘 다 돌려서 비교한다."""
    text_img = render_text_image(original_number, font_size=48, margin=10)
    comp = composite_text_with_watermark(
        text_img, wm_text="PROTECTED", angle_deg=10, gray_value=220, x_shift=10,
    )
    composite_img = comp["composite"]
    wm_mask_np = comp["wm_mask"]
    text_mask_np = comp["text_mask"]

    original_tensor = wrapper.image_to_tensor(composite_img, max_width=NUMBER_MAX_WIDTH)
    wm_mask_tensor = wrapper.mask_to_tensor(wm_mask_np, composite_img.size, max_width=NUMBER_MAX_WIDTH)
    text_mask_tensor = wrapper.mask_to_tensor(text_mask_np, composite_img.size, max_width=NUMBER_MAX_WIDTH)

    before_pred = wrapper.predict_text(original_tensor)
    log(f"  [진단] 워터마크 합성 후 OCR 베이스라인 예측: '{before_pred}' "
        f"(원본과 일치: {before_pred == original_number})")

    # (a) 실제 제약: 워터마크 영역(wm_mask)에만 섭동
    masked_result = None
    masked_eps_used = None
    for eps in EPS_CANDIDATES:
        log(f"  [진단-(a) 워터마크 영역만] eps={eps} 시도 중...")
        result = attack_watermark_region(
            ocr_wrapper=wrapper,
            original_tensor=original_tensor,
            wm_mask_tensor=wm_mask_tensor,
            target_text=target_number,
            eps=eps,
            eps_iter=eps / 10,
            max_iter=MAX_ITER_MASKED,
            verbose=False,
        )
        log(f"    -> success={result['success']}, final='{result['final_prediction']}', "
            f"final_loss={result['history'][-1][1]:.3f}")
        masked_result = result
        masked_eps_used = eps
        if result["success"]:
            break

    # 사람 가독성 확인 (text_mask 영역이 안 바뀌었는지)
    diff = (masked_result["adv_tensor"] - original_tensor).abs()
    max_text_diff = (diff * text_mask_tensor).max().item()

    # (b) 진단용: 제약 없이 전체 이미지(텍스트 픽셀 포함) 공격
    full_mask_tensor = torch.ones_like(original_tensor)
    unmasked_result = attack_watermark_region(
        ocr_wrapper=wrapper,
        original_tensor=original_tensor,
        wm_mask_tensor=full_mask_tensor,
        target_text=target_number,
        eps=0.6,
        eps_iter=0.06,
        max_iter=MAX_ITER_UNMASKED,
        verbose=False,
    )
    log(f"  [진단-(b) 제약 없음, 가독성 무시] success={unmasked_result['success']}, "
        f"final='{unmasked_result['final_prediction']}', "
        f"success_iter={unmasked_result['success_iter']}")

    return {
        "before_pred": before_pred,
        "masked_eps_used": masked_eps_used,
        "masked_success": masked_result["success"],
        "masked_final_prediction": masked_result["final_prediction"],
        "masked_final_loss": masked_result["history"][-1][1],
        "text_region_max_diff": max_text_diff,
        "unmasked_success": unmasked_result["success"],
        "unmasked_success_iter": unmasked_result["success_iter"],
        "masked_adv_tensor": masked_result["adv_tensor"],
        "original_tensor": original_tensor,
    }


# ---------------------------------------------------------------------------
# 이미지 생성 (Claude에게 보여줄 최종 "방정식+선택지" 이미지)
# ---------------------------------------------------------------------------
def stack_images(top_img, bottom_img, pad=24, gap=18):
    from PIL import Image
    top_img = top_img.convert("RGB")
    bottom_img = bottom_img.convert("RGB")
    w = max(top_img.width, bottom_img.width) + 2 * pad
    h = top_img.height + gap + bottom_img.height + 2 * pad
    canvas = Image.new("RGB", (w, h), color=(255, 255, 255))
    canvas.paste(top_img, ((w - top_img.width) // 2, pad))
    canvas.paste(bottom_img, ((w - bottom_img.width) // 2, pad + top_img.height + gap))
    return canvas


def build_question_image(equation_text, options_text):
    """선택지와 방정식을 '실제 글자 그대로' 렌더링한 깔끔한 이미지 (베이스라인/치환용)."""
    eq_img = render_text_image(equation_text, font_size=48, margin=14)
    opt_img = render_text_image(options_text, font_size=36, margin=14)
    return stack_images(eq_img, opt_img)


def concat_horizontal(images, gap=4, pad=14):
    """이미지들을 가로로 이어붙인다 (세로 중앙 정렬). "3x + 2 = " + 공격당한
    숫자 이미지를 하나의 방정식 줄로 재구성할 때 쓴다."""
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


# ---------------------------------------------------------------------------
# Claude(Timely API, Anthropic Messages 형식) 호출
# ---------------------------------------------------------------------------
def ask_claude(pil_image):
    image_b64 = encode_pil_image_to_base64(pil_image)
    # max_tokens=50은 부족했다: 이 모델은 답변 전에 thinking(내부 추론) 블록을
    # 먼저 쓰는데, 50 토큰이 거기서 다 소모되면 실제 "정답: X" 텍스트가 전혀
    # 안 나오고 빈 응답이 된다 (stop_reason="max_tokens"). 500으로 넉넉히 둔다.
    return timely_ask_claude(text=PROMPT_TEMPLATE, image_b64=image_b64, max_tokens=500)


def parse_answer_letter(response_text):
    match = re.search(r"정답\s*[:：]?\s*([A-Da-d])", response_text)
    if match:
        return match.group(1).upper()
    fallback = re.search(r"\b([A-Da-d])\b", response_text)
    return fallback.group(1).upper() if fallback else None


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main():
    from PIL import Image

    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    if not api_key_is_configured():
        log(f"[오류] TIMELY_API_KEY가 설정되지 않았습니다 ({ENV_PATH}). Claude 호출 단계는 건너뜁니다.")
        api_available = False
    else:
        api_available = True

    log("[info] EasyOCR Reader 로딩 중 (quantize=False, 공격용 설정)...")
    import easyocr
    reader = easyocr.Reader(["ko", "en"], gpu=False, quantize=False)
    wrapper = OcrAttackModel(reader)

    all_results = []

    for q in QUESTIONS:
        log(f"\n{'='*70}\n[문항 {q['id']}] {q['equation_template'].format(q['original_number'])} "
            f"-> 목표: {q['equation_template'].format(q['target_number'])}\n{'='*70}")

        # ---- 1단계: OCR 표적 공격 진단 ----
        diag = diagnose_ocr_targeted_attack(
            wrapper, q["original_number"], q["target_number"], log
        )

        # ---- 2단계: Claude 테스트용 이미지 3종 ----
        baseline_eq = q["equation_template"].format(q["original_number"])
        substituted_eq = q["equation_template"].format(q["target_number"])

        baseline_img = build_question_image(baseline_eq, q["options"])
        substituted_img = build_question_image(substituted_eq, q["options"])

        baseline_path = os.path.join(OUTPUT_DIR, f"{q['id']}_baseline.png")
        substituted_path = os.path.join(OUTPUT_DIR, f"{q['id']}_substituted.png")
        baseline_img.save(baseline_path)
        substituted_img.save(substituted_path)

        # 워터마크 시도 이미지 (실패했더라도 기록용으로 생성)
        # 주의: 공격은 숫자 부분("14" 등)만 떼어내서 수행했으므로, Claude에게
        # 보여줄 때는 방정식 앞부분("3x + 2 = " 등)을 다시 붙여서 베이스라인과
        # 똑같은 맥락(선택지를 풀 수 있는 완전한 방정식)으로 맞춰야 공정한
        # 비교가 된다. (처음 버전은 숫자만 덜렁 보여줘서 Claude가 방정식 자체를
        # 볼 수 없었던 버그가 있었음)
        prefix_text = q["equation_template"].split("{}")[0]
        prefix_img = render_text_image(prefix_text, font_size=48, margin=10)

        number_adv_img = tensor_to_pil_image(diag["masked_adv_tensor"])
        # prefix_img와 높이를 맞춰서 자연스러운 한 줄 방정식처럼 보이게 함
        scale = prefix_img.height / number_adv_img.height
        number_adv_img = number_adv_img.resize(
            (max(1, int(number_adv_img.width * scale)), prefix_img.height), Image.NEAREST
        )
        equation_line_img = concat_horizontal([prefix_img, number_adv_img], gap=2, pad=14)

        options_img = render_text_image(q["options"], font_size=36, margin=14)
        watermark_attempt_img = stack_images(equation_line_img, options_img)
        watermark_path = os.path.join(OUTPUT_DIR, f"{q['id']}_watermark_attempt.png")
        watermark_attempt_img.save(watermark_path)

        q_result = {
            "id": q["id"],
            "baseline_equation": baseline_eq,
            "target_equation": substituted_eq,
            "original_answer": q["original_answer"],
            "target_answer": q["target_answer"],
            "ocr_diagnosis": {
                "masked_eps_used": diag["masked_eps_used"],
                "masked_success": diag["masked_success"],
                "masked_final_prediction": diag["masked_final_prediction"],
                "text_region_max_diff": diag["text_region_max_diff"],
                "unmasked_success": diag["unmasked_success"],
                "unmasked_success_iter": diag["unmasked_success_iter"],
            },
        }

        if api_available:
            log(f"  [Claude 호출] (a) 베이스라인 이미지...")
            try:
                resp_a = ask_claude(baseline_img)
                ans_a = parse_answer_letter(resp_a)
            except Exception as e:
                resp_a, ans_a = f"[오류] {e}", None
            log(f"    응답: {resp_a.strip()[:80]!r} -> 파싱된 답: {ans_a}")
            time.sleep(0.5)

            log(f"  [Claude 호출] (b) 숫자 실제 치환 이미지 (사니티 체크)...")
            try:
                resp_b = ask_claude(substituted_img)
                ans_b = parse_answer_letter(resp_b)
            except Exception as e:
                resp_b, ans_b = f"[오류] {e}", None
            log(f"    응답: {resp_b.strip()[:80]!r} -> 파싱된 답: {ans_b}")
            time.sleep(0.5)

            log(f"  [Claude 호출] (c) 워터마크 시도 이미지 (OCR 표적 실패 상태 포함)...")
            try:
                resp_c = ask_claude(watermark_attempt_img)
                ans_c = parse_answer_letter(resp_c)
            except Exception as e:
                resp_c, ans_c = f"[오류] {e}", None
            log(f"    응답: {resp_c.strip()[:80]!r} -> 파싱된 답: {ans_c}")

            q_result["claude"] = {
                "baseline_response": resp_a.strip(),
                "baseline_answer": ans_a,
                "baseline_correct": ans_a == q["original_answer"],
                "substituted_response": resp_b.strip(),
                "substituted_answer": ans_b,
                "substituted_hit_target": ans_b == q["target_answer"],
                "watermark_response": resp_c.strip(),
                "watermark_answer": ans_c,
                "watermark_hit_target": ans_c == q["target_answer"],
            }

        all_results.append(q_result)

    # ---- 결과 저장 및 요약 ----
    result_path = os.path.join(OUTPUT_DIR, "pilot_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    log(f"\n{'='*70}\n[전체 요약]\n{'='*70}")
    for r in all_results:
        log(f"\n[{r['id']}] {r['baseline_equation']} (정답 {r['original_answer']}) "
            f"-> 목표 오답 {r['target_answer']} (목표 식: {r['target_equation']})")
        d = r["ocr_diagnosis"]
        log(f"  OCR 진단: 워터마크 영역만 공격 성공={d['masked_success']} "
            f"(최종 예측 '{d['masked_final_prediction']}', eps={d['masked_eps_used']}), "
            f"제약 없이 공격 성공={d['unmasked_success']} "
            f"(iter={d['unmasked_success_iter']}), "
            f"글자영역 최대변화={d['text_region_max_diff']:.6f}")
        if "claude" in r:
            c = r["claude"]
            log(f"  Claude: 베이스라인 정답여부={c['baseline_correct']} (답={c['baseline_answer']}), "
                f"실제치환시 목표오답 적중={c['substituted_hit_target']} (답={c['substituted_answer']}), "
                f"워터마크시도시 목표오답 적중={c['watermark_hit_target']} (답={c['watermark_answer']})")

    log(f"\n[결과 저장] {result_path}")
    log(f"[이미지 저장 위치] {OUTPUT_DIR}")

    log_path = os.path.join(OUTPUT_DIR, "pilot_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))


if __name__ == "__main__":
    main()
