# -*- coding: utf-8 -*-
"""
Timely API(신버전, Anthropic Messages 형식) 호출 공통 헬퍼.

배경 (2026-11-25):
- 기존 "OpenAI 호환 모드"(.../bridge/openai)는 2026-11-18에 폐지되어 404/410만
  반환한다. Timely가 "Timely API"(direct 모드)로 전환했고, Claude는 OpenAI
  Chat Completions 형식이 아니라 Anthropic Messages API 형식(/v1/messages)으로
  호출해야 한다.
  참고: https://github.com/timely-hub/timely-gpt-sdk/blob/master/DIRECT_API_GUIDE.md
- 이미지 입력 형식도 바뀌었다: OpenAI 방식의
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
  대신 Anthropic 방식의
    {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                   "data": "..."}}
  를 쓴다.
- 인증 키 형식도 바뀌었다 (tgpt_sk_... 형식의 새 "연동 키" 필요, 기존 키는
  새 시스템에서 401 "유효하지 않은 API 키입니다"로 거절됨).

보안 주의: 이 모듈은 .env에서 키를 읽어오지만, 키 값 자체를 출력하거나
로그에 남기지 않는다.
"""
import base64
import os

import requests
from dotenv import load_dotenv

ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(ENV_PATH)

API_KEY = os.environ.get("TIMELY_API_KEY")
BASE_URL = os.environ.get("TIMELY_BASE_URL", "https://hello.timelyai.io/api/v2/chat/direct/anthropic")
MODEL = os.environ.get("TIMELY_MODEL", "claude-sonnet-5")
ANTHROPIC_VERSION = "2023-06-01"


def api_key_is_configured():
    """키가 실제로 설정되어 있는지 확인한다 (값 자체는 반환하지 않음)."""
    return bool(API_KEY) and API_KEY.strip() != "" and "여기에" not in API_KEY


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def encode_pil_image_to_base64(pil_image):
    import io
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def ask_claude(text, image_path=None, image_b64=None, max_tokens=60, model=None, timeout=60, thinking=False):
    """Claude(Timely API, Anthropic Messages 형식)에 텍스트(+선택적 이미지)를 보내고
    응답 텍스트를 반환한다.

    비용 메모 (2026-11-25): 이 모델은 복잡한 질문(예: 방정식 풀이)에 기본적으로
    "thinking"(내부 추론) 블록을 먼저 쓰는데, 거기서 토큰을 많이 써버리면
    max_tokens가 작을 때 실제 답변 텍스트가 전혀 안 나올 수 있다 (앞서
    pilot_targeted에서 실제로 겪은 문제). thinking=False(기본값)로 두면
    thinking을 꺼서 max_tokens를 훨씬 작게(60 정도) 써도 안전하고, 블랙박스
    탐색처럼 호출을 많이 해야 할 때 비용을 크게 줄인다. 모델이 더 깊은 추론을
    하길 원하면 thinking=True로 켜고 max_tokens도 충분히(300~500) 키워야 한다.

    Args:
        text: 사용자 질문 텍스트.
        image_path: 첨부할 이미지 파일 경로 (image_b64와 동시에 쓸 수 없음).
        image_b64: 이미 base64로 인코딩된 이미지 (PIL 이미지를 바로 보낼 때 유용).
        max_tokens: 응답 최대 토큰 수. thinking=False면 60 정도로도 충분하다.
        model: 모델명 (None이면 .env의 TIMELY_MODEL, 보통 "claude-sonnet-5").
        timeout: 요청 타임아웃(초).
        thinking: True면 내부 추론(thinking)을 허용 (토큰을 더 많이 쓰고
            max_tokens도 크게 줘야 함). False(기본값)면 비활성화해서 비용을 줄임.

    Returns:
        응답 텍스트 (여러 블록이면 이어붙임).

    Raises:
        RuntimeError: 키가 설정되지 않은 경우.
        requests.exceptions.HTTPError: HTTP 오류 응답인 경우.
    """
    if not api_key_is_configured():
        raise RuntimeError(
            f"TIMELY_API_KEY가 설정되지 않았습니다. {ENV_PATH} 파일을 확인해주세요."
        )

    content = []
    if image_path is not None:
        image_b64 = encode_image_to_base64(image_path)
    if image_b64 is not None:
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": image_b64},
        })
    content.append({"type": "text", "text": text})

    payload = {
        "model": model or MODEL,
        "max_tokens": max_tokens,
        "thinking": {"type": "enabled", "budget_tokens": max(1024, max_tokens)} if thinking else {"type": "disabled"},
        "messages": [{"role": "user", "content": content}],
    }
    headers = {
        "x-api-key": API_KEY,
        "anthropic-version": ANTHROPIC_VERSION,
        "Content-Type": "application/json",
    }
    url = f"{BASE_URL.rstrip('/')}/v1/messages"

    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()

    blocks = data.get("content", [])
    texts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
    return "\n".join(texts)
