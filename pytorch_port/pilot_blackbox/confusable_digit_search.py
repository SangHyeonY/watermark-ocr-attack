# -*- coding: utf-8 -*-
"""
방향 3(블랙박스 탐색), 새 변수: "진짜 헷갈리는 두 숫자 모양을 섞기"

지금까지(FAWA 노이즈, 숨은 글자, QR)는 전부 "특정 방법의 허점을 찔러서 AI만
속이기"였는데 다 실패했다. 이번에는 완전히 다른 성격의 변수를 쓴다:
- 노이즈가 아니라, 실제로 모양이 비슷한 두 숫자("1"과 "7")를 알파 블렌딩으로
  섞어서, "진짜 시각적으로 애매해지는 지점"이 어디인지 찾는다.
- 이건 특정 모델(EasyOCR)의 좁은 허점을 노리는 게 아니라, 사람도 애매해할 수
  있는 진짜 모호함이다. 그래서 핵심 질문은 "AI가 헷갈리기 시작하는 비율"과
  "사람이 헷갈리기 시작하는 비율"이 다른지(= 그 사이에 쓸 수 있는 틈이 있는지)이다.

비용 절감:
- 질문을 아주 단순하게("숫자가 몇이야?") 만들어서 thinking을 끄고 max_tokens를
  작게 유지한다.
- 7단계만 테스트한다 (총 7회 호출).

실행 방법:
    conda activate fawa
    python pytorch_port/pilot_blackbox/confusable_digit_search.py
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from watermark import resolve_font
from timely_client import ask_claude, api_key_is_configured, encode_pil_image_to_base64, ENV_PATH

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "confusable_digit")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CANVAS_SIZE = (150, 150)
FONT_SIZE = 100
DIGIT_A = "1"
DIGIT_B = "7"
ALPHAS = [0.0, 0.15, 0.3, 0.45, 0.5, 0.6, 0.75, 1.0]  # 0.0=순수 "1", 1.0=순수 "7"

PROMPT = "이 이미지에 숫자가 하나 보입니다. 숫자가 몇인지 숫자만 답해주세요."


def render_digit_canvas(digit, canvas_size=CANVAS_SIZE, font_size=FONT_SIZE):
    """숫자를 고정 캔버스 중앙에 렌더링한 흑백(L) numpy 배열로 반환."""
    font_path = resolve_font()
    font = ImageFont.truetype(font_path, font_size)
    img = Image.new("L", canvas_size, color=255)
    draw = ImageDraw.Draw(img)
    bbox = font.getbbox(digit)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (canvas_size[0] - w) // 2 - bbox[0]
    y = (canvas_size[1] - h) // 2 - bbox[1]
    draw.text((x, y), digit, fill=0, font=font)
    return np.array(img).astype(np.float32)


def blend(arr_a, arr_b, alpha):
    blended = (1 - alpha) * arr_a + alpha * arr_b
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="L")


def main():
    if not api_key_is_configured():
        print(f"[오류] TIMELY_API_KEY가 설정되지 않았습니다 ({ENV_PATH}).")
        sys.exit(1)

    arr_a = render_digit_canvas(DIGIT_A)
    arr_b = render_digit_canvas(DIGIT_B)

    print(f"[info] '{DIGIT_A}' <-> '{DIGIT_B}' 블렌딩, {len(ALPHAS)}단계 테스트 (thinking 비활성화, 저비용)\n")

    results = []
    for alpha in ALPHAS:
        img = blend(arr_a, arr_b, alpha).convert("RGB")
        img_path = os.path.join(OUTPUT_DIR, f"alpha_{alpha:.2f}.png")
        img.save(img_path)

        image_b64 = encode_pil_image_to_base64(img)
        resp = ask_claude(text=PROMPT, image_b64=image_b64, max_tokens=20, thinking=False)
        resp_clean = resp.strip()

        print(f"  alpha={alpha:.2f} ('{DIGIT_A}' {100*(1-alpha):.0f}% / '{DIGIT_B}' {100*alpha:.0f}%) "
              f"-> Claude 응답: {resp_clean!r}")
        results.append({"alpha": alpha, "response": resp_clean, "image_path": img_path})

    print(f"\n{'='*60}")
    print(f"[요약] alpha(={DIGIT_B} 비율)에 따른 Claude 응답")
    print(f"{'='*60}")
    for r in results:
        print(f"  alpha={r['alpha']:.2f} -> {r['response']}")

    print(f"\n[이미지 저장 위치] {OUTPUT_DIR}")
    print("[다음 단계] 어느 alpha부터 Claude가 '7'이라고 답하는지 확인한 뒤, "
          "그 근처 이미지들을 사람이 직접 보고 '1'인지 '7'인지 구분되는지 비교해야 함")


if __name__ == "__main__":
    main()
