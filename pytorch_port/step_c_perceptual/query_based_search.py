# -*- coding: utf-8 -*-
"""
Step C-Perceptual (4차): 쿼리 기반(피드백 기반) 저대비 탐색.

배경:
- 1~3차 파일럿은 전부 "한 번 찍어서" 만든 패턴을 테스트했다 (노이즈 강도,
  저대비 단계를 전부 사전에 고정값으로 정해놓고 결과만 사후 확인).
- 이번에는 Claude의 실제 응답을 매 반복마다 피드백으로 사용해, 글자 밝기를
  조금씩 조정해나가는 방식을 시도한다. 다만 비용 문제 때문에 SPSA처럼
  매 반복 2회(+delta, -delta) 호출하는 방식은 쓰지 않고, 한 방향으로만
  단조 증가시키며 1회씩만 호출하는 가장 단순한 "언덕 오르기" 방식을 쓴다.

세밀한 신호 확보:
- 이전 파일럿(measure_ai_accuracy.py 등)은 "정답을 맞혔는지"(O/X)만 봤는데,
  이는 신호가 너무 거칠어서 작은 변화가 묻힌다. 이번에는 "이 선택지를 글자
  그대로 읽어줘"라고 시켜서, 읽어낸 텍스트와 정답 텍스트 사이의 CER(문자
  오류율, text_metrics.py)을 계산한다. 0.0(완전히 맞음)~1.0+(완전히 틀림)
  사이의 연속값이라 O/X보다 훨씬 민감하게 변화를 포착할 수 있다.

비용 절감을 위한 설계:
- 문항 1개, 선택지 1개만 사용 (전체 문제가 아니라 "B) 세종대왕"같은 짧은
  텍스트 한 줄만 렌더링)
- 글자 밝기(gray_level)를 10씩 증가시키며 1회씩만 호출 (최대 20회 제한)
- CER이 유의미하게 오르기 시작하는 지점을 찾으면 조기 종료

실행 방법:
    conda activate fawa
    python pytorch_port/step_c_perceptual/query_based_search.py
"""
import base64
import os
import sys
import time

import requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from watermark import resolve_font  # noqa: E402
from text_metrics import compute_cer  # noqa: E402

ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(ENV_PATH)

API_KEY = os.environ.get("TIMELY_API_KEY")
BASE_URL = os.environ.get("TIMELY_BASE_URL", "https://hello.timelygpt.co.kr/api/v2/chat/bridge/openai")
MODEL = os.environ.get("TIMELY_MODEL", "anthropic/claude-sonnet-5")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "query_based")

# 테스트 대상: q02의 정답 선택지 "세종대왕" (한글 4음절, 적당한 길이)
CHOICE_LABEL = "B)"
CHOICE_TEXT = "세종대왕"

FONT_SIZE = 48
MARGIN = 20

MAX_CALLS = 10  # 1차로 작게 시작 (사용자 요청: 10회 정도로 먼저 확인)
GRAY_START = 0
GRAY_STEP = 20  # 0 -> 20 -> 40 ... 단계로 증가 (10회면 0~180까지 커버)

READ_PROMPT_TEMPLATE = (
    "이 이미지에 적힌 글자를 그대로 읽어서 알려주세요. "
    "다른 설명 없이 읽은 텍스트만 그대로 답하세요."
)


def render_choice_line(label, text, gray_level, font_path=None):
    """선택지 한 줄(예: "B) 세종대왕")만 지정한 밝기로 렌더링한다."""
    font_path = resolve_font(font_path)
    font = ImageFont.truetype(font_path, FONT_SIZE)
    line = f"{label} {text}"
    text_color = (gray_level, gray_level, gray_level)

    bbox = font.getbbox(line)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    img_w, img_h = w + 2 * MARGIN, h + 2 * MARGIN

    img = Image.new("RGB", (img_w, img_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((MARGIN - bbox[0], MARGIN - bbox[1]), line, fill=text_color, font=font)
    return img


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def ask_model_to_read(image_path):
    image_b64 = encode_image_to_base64(image_path)
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": READ_PROMPT_TEMPLATE},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ],
        }],
        "max_tokens": 50,
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    url = f"{BASE_URL.rstrip('/')}/chat/completions"
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def main():
    if not API_KEY or API_KEY.strip() == "" or "여기에" in API_KEY:
        print("[오류] TIMELY_API_KEY가 설정되지 않았습니다.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ground_truth = f"{CHOICE_LABEL} {CHOICE_TEXT}"
    print(f"[info] 대상 텍스트: '{ground_truth}'")
    print(f"[info] 글자 밝기를 {GRAY_START}부터 {GRAY_STEP}씩 올리며 최대 {MAX_CALLS}회 호출합니다.\n")

    history = []
    for i in range(MAX_CALLS):
        gray_level = GRAY_START + i * GRAY_STEP
        if gray_level >= 255:
            print(f"[중단] gray_level={gray_level}은 배경색(255)과 같아져 글자가 사라집니다.")
            break

        img = render_choice_line(CHOICE_LABEL, CHOICE_TEXT, gray_level)
        image_path = os.path.join(OUTPUT_DIR, f"step{i:02d}_gray{gray_level}.png")
        img.save(image_path)

        try:
            response_text = ask_model_to_read(image_path)
        except Exception as e:
            print(f"  [{i}] gray={gray_level}: [오류] {type(e).__name__}: {e}")
            break

        cer = compute_cer(response_text.strip(), ground_truth)
        history.append({"step": i, "gray_level": gray_level, "response": response_text.strip(), "cer": cer})

        print(f"  [{i}] gray={gray_level:3d} (배경과 차이={255-gray_level:3d}) "
              f"-> 응답: '{response_text.strip()}' | CER={cer:.3f}")

        time.sleep(0.5)

    print(f"\n{'='*60}")
    print("[요약] gray_level 증가에 따른 CER 추이")
    print(f"{'='*60}")
    for h in history:
        bar = "#" * int(h["cer"] * 40)
        print(f"  gray={h['gray_level']:3d}  CER={h['cer']:.3f}  {bar}")

    max_cer_entry = max(history, key=lambda h: h["cer"]) if history else None
    if max_cer_entry and max_cer_entry["cer"] > 0:
        print(f"\n[최대 CER] gray_level={max_cer_entry['gray_level']}에서 CER={max_cer_entry['cer']:.3f} "
              f"(응답: '{max_cer_entry['response']}')")
    else:
        print("\n[결과] 모든 단계에서 CER=0.0 (전혀 흔들리지 않음)")


if __name__ == "__main__":
    main()
