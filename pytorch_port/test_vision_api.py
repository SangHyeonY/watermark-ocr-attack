# -*- coding: utf-8 -*-
"""
Timely API(Anthropic Messages 형식)로 이미지 입력(vision)이 실제로 되는지 확인하는
작은 테스트 스크립트.

목적:
- 작업정리 문서의 "이미지 입력 AI 최소 1종 선정" 요건을 위해, 학교에서 발급받은
  Timely API 키로 이미지를 첨부한 질문이 실제로 동작하는지 먼저 확인한다.
- Step B(20문항 베이스라인 측정)를 설계하기 전에, 이 연결이 제대로 되는지
  작은 단위로 먼저 검증하는 단계.

[2026-11-25 갱신] 기존 "OpenAI 호환 모드"(.../bridge/openai)가 2026-11-18에
폐지되어, Timely API(direct 모드, Anthropic Messages 형식)로 전면 교체했다.
호출 로직은 timely_client.py 공통 헬퍼로 옮겼다. 자세한 배경은
timely_client.py 모듈 docstring과 .env.example 참고.

보안 주의:
- 이 스크립트는 .env 파일에서 API 키를 읽어오며, 키 값 자체를 출력하거나
  로그에 남기지 않는다. (파일 상단 .env.example 참고)
- .env 파일은 .gitignore에 등록되어 git에 올라가지 않는다.

실행 방법:
    conda activate fawa
    python pytorch_port/test_vision_api.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from timely_client import ask_claude, api_key_is_configured, ENV_PATH, MODEL

# 테스트에 사용할 이미지 (5단계에서 만든 결과물 중 하나를 재사용)
TEST_IMAGE_PATH = os.path.join(
    os.path.dirname(__file__), "outputs", "watermark_test", "문제_original.png"
)


def main():
    if not api_key_is_configured():
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

    try:
        answer = ask_claude(
            text="이 이미지에 어떤 한글 단어가 보이나요? 단어만 정확히 답해주세요.",
            image_path=TEST_IMAGE_PATH,
            max_tokens=100,
        )
    except Exception as e:
        print(f"[오류] 요청 실패: {type(e).__name__}: {e}")
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
