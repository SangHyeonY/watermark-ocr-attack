# -*- coding: utf-8 -*-
"""
ocr_model.py (OcrAttackModel) 검증 스크립트.

확인할 것:
1. 전처리(image_to_tensor)가 올바른지 - predict_text가 원본 글자를 정확히 맞추는가.
2. ctc_loss가 정상적으로 역전파(backward)되어 입력 이미지 텐서에 gradient가 생기는가.
3. (quantize=False가 왜 필요한지) quantize=True로 만든 Reader로는 gradient가
   안 나온다는 것도 같이 보여준다 (대조 확인).

실행 방법:
    conda activate fawa
    python pytorch_port/test_ocr_model.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import easyocr

from watermark import render_text_image
from ocr_model import OcrAttackModel

TEST_WORDS = ["시험", "문제", "정답", "학교", "한국어"]


def main():
    print("[info] EasyOCR Reader 로딩 중 (quantize=False, 공격용 설정)...")
    reader = easyocr.Reader(["ko", "en"], gpu=False, quantize=False)
    wrapper = OcrAttackModel(reader)

    print("\n--- 1) predict_text 정확도 확인 ---")
    all_ok = True
    for word in TEST_WORDS:
        img = render_text_image(word, font_size=48, margin=10)
        tensor = wrapper.image_to_tensor(img)
        predicted = wrapper.predict_text(tensor)
        match = predicted == word
        all_ok &= match
        print(f"  [{'OK' if match else 'MISMATCH'}] 정답: '{word}' / 예측: '{predicted}'")

    print("\n--- 2) CTC loss + backward로 gradient 확인 ---")
    word = "문제"
    target = "가나다"  # 원본과 전혀 다른 임의의 목표 문자열 (공격 목표 예시)
    img = render_text_image(word, font_size=48, margin=10)
    tensor = wrapper.image_to_tensor(img)

    loss_to_target = wrapper.ctc_loss(tensor, target_text=target)
    loss_to_target.backward()
    grad_exists = tensor.grad is not None
    grad_sum = tensor.grad.abs().sum().item() if grad_exists else 0.0
    print(f"  목표 문자열: '{target}', CTC loss: {loss_to_target.item():.4f}")
    print(f"  gradient 존재 여부: {grad_exists}, gradient 절댓값 합: {grad_sum:.4f}")
    grad_ok = grad_exists and grad_sum > 0
    all_ok &= grad_ok

    print("\n--- 3) 대조 확인: quantize=True (기본값)로 만들면 gradient가 안 나옴 ---")
    reader_q = easyocr.Reader(["ko", "en"], gpu=False, quantize=True)
    wrapper_q = OcrAttackModel(reader_q)
    tensor_q = wrapper_q.image_to_tensor(img)
    try:
        loss_q = wrapper_q.ctc_loss(tensor_q, target_text=target)
        loss_q.backward()
        grad_q_exists = tensor_q.grad is not None
    except Exception as e:
        grad_q_exists = False
        print(f"  (quantize=True에서는 forward/backward 중 예외 발생: {type(e).__name__})")
    print(f"  quantize=True일 때 gradient 존재 여부: {grad_q_exists} (예상: False)")

    print(f"\n[최종 결과] {'성공' if all_ok else '실패'}")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
