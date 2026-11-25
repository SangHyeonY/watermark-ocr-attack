# -*- coding: utf-8 -*-
"""
QR코드 인코딩 아이디어의 1차 사니티 체크: "Claude가 QR코드를 픽셀만으로
디코딩할 수 있는가?"

배경:
- 숨은 연한 텍스트(명암 기반 은닉)는 실패했다. 명암이 아니라 "해독 불가능함"
  자체로 숨기는 방식(QR코드/바코드형 패턴)이 통하는지 먼저 확인해야 한다.
- QR코드 안에 목표 숫자("17")나 목표 정답("정답: C")을 인코딩해두고, Claude에게
  "이 QR코드에 뭐라고 써있어?"라고 직접 물어본다.
- 만약 Claude가 QR을 정확히 디코딩하지 못하면, 그 다음 단계(QR을 문제 이미지
  한쪽에 끼워넣어 전이 테스트)는 의미가 없으므로 여기서 멈춘다.

결과 (기록용, pilot_blackbox/outputs/qr_sanity/qr_sanity_result.json 참고):
- Claude Sonnet 5는 QR코드 디코딩을 거절했다. "픽셀을 보는 것만으로 정확히
  디코딩할 수는 없다"며 QR 스캐너 앱을 쓰라고 안내함. 즉 디코딩 능력 자체가
  없는 것으로 판단, QR코드 기반 아이디어는 여기서 중단.

실행 방법:
    conda activate fawa  (또는 ~/miniconda3/bin/activate fawa)
    pip install qrcode   (처음 1회만, 설치 안 돼 있으면)
    python pytorch_port/pilot_blackbox/qr_sanity_check.py
"""
import json
import os
import sys

import qrcode

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from timely_client import ask_claude, api_key_is_configured, encode_pil_image_to_base64, ENV_PATH

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "qr_sanity")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_CASES = [
    {"id": "number", "encoded_text": "17"},
    {"id": "direct_answer", "encoded_text": "정답: C"},
]

PROMPT = "이 QR코드를 디코딩해서 그 안에 뭐라고 써 있는지만 답해줘."


def main():
    if not api_key_is_configured():
        print(f"[오류] TIMELY_API_KEY가 설정되지 않았습니다 ({ENV_PATH}).")
        sys.exit(1)

    results = []
    for case in TEST_CASES:
        img = qrcode.make(case["encoded_text"], box_size=8, border=2)
        img_path = os.path.join(OUTPUT_DIR, f"qr_{case['id']}.png")
        img.save(img_path)

        image_b64 = encode_pil_image_to_base64(img.convert("RGB"))
        # QR 디코딩 여부를 확인하는 단순 질문이라 thinking 불필요 (비용 절감)
        resp = ask_claude(text=PROMPT, image_b64=image_b64, max_tokens=150, thinking=False)

        decoded_correctly = case["encoded_text"] in resp
        print(f"[{case['id']}] 인코딩한 내용: '{case['encoded_text']}'")
        print(f"  Claude 응답: {resp.strip()[:200]!r}")
        print(f"  디코딩 성공 여부(단순 포함 검사): {decoded_correctly}\n")

        results.append({
            "id": case["id"], "encoded_text": case["encoded_text"],
            "image_path": img_path, "response": resp.strip(),
            "decoded_correctly": decoded_correctly,
        })

    result_path = os.path.join(OUTPUT_DIR, "qr_sanity_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[결과 저장] {result_path}")
    success_count = sum(1 for r in results if r["decoded_correctly"])
    print(f"[종합] {success_count}/{len(results)}개 QR코드를 정확히 디코딩함")
    if success_count == 0:
        print("[결론] Claude는 QR코드를 픽셀만으로 디코딩하지 못함 -> QR 기반 아이디어 중단")


if __name__ == "__main__":
    main()
