# -*- coding: utf-8 -*-
"""
FAWA 공격 전체 파이프라인 엔드투엔드 테스트 (untargeted 방식).

흐름:
1. 한글 단어 이미지를 만든다 (원본 글자).
2. 그 위에 워터마크를 합성한다 (watermark.py) -> wm_mask, text_mask 얻음.
3. 합성 이미지와 wm_mask를 OCR 모델 입력 텐서로 변환한다 (ocr_model.py).
4. wm_mask 영역에만 적대적 섭동을 가해서, OCR이 더 이상 원래 글자를 맞히지
   못하게 만든다 (attack.py의 attack_untargeted - "특정 오답으로 유도"가 아니라
   "정답을 못 맞히게만 하면 성공"인 더 쉬운 목표. WORK_LOG.md 5단계 참고).
5. 공격 성공 여부, 그리고 "text_mask 영역(사람이 읽는 실제 글자)이 공격 후에도
   전혀 바뀌지 않았는지"를 확인한다 - 이게 FAWA의 핵심 약속
   ("사람 가독성 유지 + AI만 속임")이기 때문에 반드시 확인해야 한다.

참고 (targeted -> untargeted 전환 이유):
- 원본 FAWA 논문/코드(wm_grad.py)는 "특정 목표 문자열로 속이기"(targeted)
  방식이었고, 1차 포팅도 이를 그대로 따랐다. 그런데 작업정리 문서의 핵심 가설은
  "AI가 정답을 못 맞히게 하면 충분하다"는 목표이고, 실험적으로도 targeted는
  eps(허용 변화폭)를 크게 늘려야 겨우 성공하는 등 비효율적이었다. 반면 untargeted는
  같은 조건에서 훨씬 쉽고 빠르게 성공해, 문서 취지와 실험 효율성 모두가 가리키는
  untargeted로 최종 전환했다.

실행 방법:
    conda activate fawa
    python pytorch_port/test_attack.py
"""
import os
import sys

import easyocr

sys.path.insert(0, os.path.dirname(__file__))
from watermark import render_text_image, composite_text_with_watermark
from ocr_model import OcrAttackModel, tensor_to_pil_image
from attack import attack_untargeted

# 테스트 단어 목록 (스모크 테스트/워터마크 테스트와 동일한 5개 단어로 일관성 유지)
TEST_WORDS = ["시험", "문제", "정답", "학교", "한국어"]

# 공격 허용 변화폭(eps). 0.3~0.6 사이 여러 값으로 비교한 결과
# (WORK_LOG.md 6단계 참고):
#   eps=0.3 -> 5개 중 3개만 성공, 0.4 -> 4개 성공, 0.45 -> 5개 전부 성공,
#   0.6 -> 5개 전부 성공하지만 눈으로 봤을 때 워터마크 변화가 더 거슬림.
# "공격 성공률 100%를 유지하는 가장 작은 값"인 0.45를 기본값으로 채택
# (사람 가독성/거슬림과 공격 성공률 사이의 균형점).
EPS = 0.45
EPS_ITER = 0.045
MAX_ITER = 150

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "attack_test")


def check_text_region_unchanged(original_tensor, adv_tensor, text_mask_tensor, tol=1e-4):
    """text_mask 영역(실제 글자)에서 원본과 공격 후 이미지의 차이가 없는지 확인."""
    diff = (adv_tensor - original_tensor).abs()
    diff_in_text_region = diff * text_mask_tensor
    max_diff = diff_in_text_region.max().item()
    return max_diff <= tol, max_diff


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[info] EasyOCR Reader 로딩 중 (quantize=False, 공격용 설정)...")
    reader = easyocr.Reader(["ko", "en"], gpu=False, quantize=False)
    wrapper = OcrAttackModel(reader)

    overall_ok = True
    summary = []

    for word in TEST_WORDS:
        print(f"\n{'='*60}")
        print(f"[공격 시작] 원본: '{word}' (untargeted, eps={EPS})")
        print(f"{'='*60}")

        # 1) 원본 글자 이미지
        text_img = render_text_image(word, font_size=48, margin=10)

        # 2) 워터마크 합성
        comp = composite_text_with_watermark(
            text_img, wm_text="PROTECTED", angle_deg=10, gray_value=220, x_shift=10,
        )
        composite_img = comp["composite"]
        wm_mask_np = comp["wm_mask"]
        text_mask_np = comp["text_mask"]

        composite_img.save(os.path.join(OUTPUT_DIR, f"{word}_before_attack.png"))

        # 3) 텐서 변환
        original_tensor = wrapper.image_to_tensor(composite_img)
        wm_mask_tensor = wrapper.mask_to_tensor(wm_mask_np, composite_img.size)
        text_mask_tensor = wrapper.mask_to_tensor(text_mask_np, composite_img.size)

        before_pred = wrapper.predict_text(original_tensor)
        print(f"[공격 전] OCR 예측: '{before_pred}' (원본과 일치: {before_pred == word})")

        # 4) 공격 실행 (untargeted: "원본 텍스트로만 안 읽히면 성공")
        result = attack_untargeted(
            ocr_wrapper=wrapper,
            original_tensor=original_tensor,
            wm_mask_tensor=wm_mask_tensor,
            original_text=word,
            eps=EPS,
            eps_iter=EPS_ITER,
            max_iter=MAX_ITER,
            verbose=True,
        )

        print(f"[공격 결과] success={result['success']}, "
              f"success_iter={result['success_iter']}, "
              f"최종 예측='{result['final_prediction']}'")

        # 4.5) 공격 후 이미지를 눈으로 비교할 수 있도록 PNG로 저장
        #      (OCR 입력 텐서는 흑백 32x100으로 작으므로, 원본도 같은 크기로
        #      변환해서 함께 저장해야 공정하게 비교할 수 있다)
        adv_img_for_view = tensor_to_pil_image(result["adv_tensor"])
        original_img_for_view = tensor_to_pil_image(original_tensor)
        adv_img_for_view.save(os.path.join(OUTPUT_DIR, f"{word}_after_attack.png"))
        original_img_for_view.save(os.path.join(OUTPUT_DIR, f"{word}_before_attack_32x100.png"))

        # 5) 사람이 읽는 글자 영역이 정말 안 바뀌었는지 확인
        text_unchanged, max_diff = check_text_region_unchanged(
            original_tensor, result["adv_tensor"], text_mask_tensor
        )
        print(f"[가독성 확인] 글자 영역 최대 변화량={max_diff:.6f} "
              f"({'유지됨' if text_unchanged else '!! 변경됨 - 문제 있음 !!'})")

        case_ok = result["success"] and text_unchanged
        overall_ok &= case_ok
        print(f"[이 케이스 결과] {'성공' if case_ok else '실패'}")

        summary.append({
            "word": word,
            "success": result["success"],
            "success_iter": result["success_iter"],
            "final_prediction": result["final_prediction"],
            "text_region_unchanged": text_unchanged,
        })

    print(f"\n{'='*60}")
    print("[전체 요약]")
    print(f"{'='*60}")
    for row in summary:
        status = "성공" if row["success"] and row["text_region_unchanged"] else "실패"
        print(
            f"  [{status}] '{row['word']}' -> '{row['final_prediction']}' "
            f"(성공 시점: iter {row['success_iter']}, 가독성 유지: {row['text_region_unchanged']})"
        )
    success_count = sum(1 for row in summary if row["success"] and row["text_region_unchanged"])
    print(f"\n[종합] {success_count}/{len(summary)} 단어 공격 성공")
    print(f"[전체 결과] {'모든 케이스 성공' if overall_ok else '일부 케이스 실패'}")

    if not overall_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
