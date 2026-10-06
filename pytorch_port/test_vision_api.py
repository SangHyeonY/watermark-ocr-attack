# -*- coding: utf-8 -*-
"""
Timely GPT 중계 서버(OpenAI 호환 API)로 이미지 입력(vision)이 실제로 되는지 확인하는
작은 테스트 스크립트.

목적:
- 작업정리 문서의 "이미지 입력 AI 최소 1종 선정" 요건을 위해, 학교에서 발급받은
  Timely API 키로 이미지를 첨부한 질문이 실제로 동작하는지 먼저 확인한다.
- Step B(20문항 베이스라인 측정)를 설계하기 전에, 이 연결이 제대로 되는지
  작은 단위로 먼저 검증하는 단계.

보안 주의:
- 이 스크립트는 .env 파일에서 API 키를 읽어오며, 키 값 자체를 출력하거나
  로그에 남기지 않는다. (파일 상단 .env.example 참고)
- .env 파일은 .gitignore에 등록되어 git에 올라가지 않는다.

실행 방법:
    conda activate fawa
    python pytorch_port/test_vision_api.py
"""
import base64
import os
import sys

import requests
from dotenv import load_dotenv

# 이 스크립트가 있는 폴더의 .env를 명시적으로 로드 (실행 위치와 무관하게 동작하도록)
ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(ENV_PATH)

API_KEY = os.environ.get("TIMELY_API_KEY")
BASE_URL = os.environ.get("TIMELY_BASE_URL", "https://hello.timelygpt.co.kr/api/v2/chat/bridge/openai")
MODEL = os.environ.get("TIMELY_MODEL", "anthropic/claude-sonnet-5")

# 테스트에 사용할 이미지 (5단계에서 만든 결과물 중 하나를 재사용)
TEST_IMAGE_PATH = os.path.join(
    os.path.dirname(__file__), "outputs", "watermark_test", "문제_original.png"
)


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def main():
    if not API_KEY or API_KEY.strip() == "" or "여기에" in API_KEY:
        print("[오류] TIMELY_API_KEY가 설정되지 않았습니다.")
        print(f"       {ENV_PATH} 파일을 확인해주세요.")
        print("       (.env.example을 복사해 .env로 만들고 실제 키 값을 입력해야 합니다)")
        sys.exit(1)

    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"[오류] 테스트 이미지가 없습니다: {TEST_IMAGE_PATH}")
        print("       먼저 pytorch_port/test_watermark.py를 실행해서 이미지를 생성해주세요.")
        sys.exit(1)

    print(f"[info] 사용 모델: {MODEL}")
    print(f"[info] 테스트 이미지: {TEST_IMAGE_PATH}")
    print("[info] API 키: 설정됨 (보안을 위해 값은 표시하지 않음)")

    image_b64 = encode_image_to_base64(TEST_IMAGE_PATH)

    # OpenAI 호환 Chat Completions API 형식 (vision 입력 포함)
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "이 이미지에 어떤 한글 단어가 보이나요? 단어만 정확히 답해주세요."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            }
        ],
        "max_tokens": 100,
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    # 주의: "/v1/chat/completions"가 아니라 "/chat/completions"가 맞는 경로.
    # Timely 중계 서버는 @ai-sdk/openai-compatible 방식이라 OpenAI 공식 SDK가
    # 자동으로 붙이는 "/v1/" 접두어 없이 바로 "/chat/completions"를 쓴다.
    # (처음에 "/v1/"을 붙여서 502 Bad Gateway가 발생했었음 - WORK_LOG.md 참고)
    url = f"{BASE_URL.rstrip('/')}/chat/completions"
    print(f"[info] 요청 전송 중: {url}")

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
    except requests.exceptions.RequestException as e:
        print(f"[오류] 요청 실패: {type(e).__name__}: {e}")
        sys.exit(1)

    print(f"[info] HTTP 상태 코드: {response.status_code}")

    if response.status_code != 200:
        print(f"[오류] 응답 실패. 본문 일부: {response.text[:500]}")
        sys.exit(1)

    data = response.json()
    try:
        answer = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        print(f"[오류] 응답 형식이 예상과 다릅니다: {e}")
        print(f"       전체 응답: {data}")
        sys.exit(1)

    print(f"\n[AI 응답] {answer}")

    expected_word = "문제"
    if expected_word in answer:
        print(f"\n[성공] 이미지 입력(vision)이 정상 동작하며, 정답('{expected_word}')이 포함되어 있습니다.")
    else:
        print(f"\n[주의] 응답은 받았지만 예상 단어('{expected_word}')가 포함되어 있지 않습니다. "
              f"모델이 이미지를 제대로 읽었는지 확인이 필요합니다.")


if __name__ == "__main__":
    main()
