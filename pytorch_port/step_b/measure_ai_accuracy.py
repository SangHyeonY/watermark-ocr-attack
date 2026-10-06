# -*- coding: utf-8 -*-
"""
Step B 베이스라인 측정 (2/2): 이미지 입력 AI(Claude Sonnet 5, Timely API) 정답률 수집.

render_questions.py로 만든 20문항 이미지를 Claude Sonnet 5(Timely API 경유)에게
하나씩 보여주고, "이미지를 보고 4지선다 문제를 풀어서 정답 번호를 고르라"고
요청한 뒤, 실제 정답과 비교해 정답률을 계산한다.

이 측정은 Step B의 "원본(보호 패턴 미적용) 베이스라인"에 해당한다. 이후 Step C에서
워터마크 적용 후의 정답률과 비교하는 기준점이 된다.

보안 주의:
- API 키는 .env 파일에서만 읽고, 코드에서 값을 출력하거나 로그에 남기지 않는다.

실행 방법:
    conda activate fawa
    python pytorch_port/step_b/measure_ai_accuracy.py
"""
import base64
import json
import os
import re
import sys
import time

import requests
from dotenv import load_dotenv

ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(ENV_PATH)

API_KEY = os.environ.get("TIMELY_API_KEY")
BASE_URL = os.environ.get("TIMELY_BASE_URL", "https://hello.timelygpt.co.kr/api/v2/chat/bridge/openai")
MODEL = os.environ.get("TIMELY_MODEL", "anthropic/claude-sonnet-5")

QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "questions.json")
IMAGE_DIR = os.path.join(os.path.dirname(__file__), "outputs", "question_images")
RESULT_PATH = os.path.join(os.path.dirname(__file__), "outputs", "ai_baseline_result.json")

CHOICE_LABELS = ["①", "②", "③", "④"]

PROMPT_TEMPLATE = (
    "이 이미지는 4지선다 객관식 문제입니다. 문제를 읽고 정답을 골라주세요.\n"
    "반드시 아래 형식으로만 답하세요 (다른 설명 없이 숫자만):\n"
    "정답: N\n"
    "(N은 1, 2, 3, 4 중 하나이며, 선택지 순서대로 1번=①, 2번=②, 3번=③, 4번=④입니다)"
)


def load_questions():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def ask_model(image_path):
    """이미지를 Claude Sonnet 5(Timely API)에 보내고 응답 텍스트를 반환한다."""
    image_b64 = encode_image_to_base64(image_path)

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT_TEMPLATE},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            }
        ],
        "max_tokens": 50,
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    url = f"{BASE_URL.rstrip('/')}/chat/completions"

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def parse_answer_number(response_text):
    """AI 응답 텍스트에서 "정답: N" 형식의 숫자를 추출한다. 못 찾으면 None."""
    match = re.search(r"정답\s*[:：]?\s*([1-4])", response_text)
    if match:
        return int(match.group(1))
    # 형식을 안 지켰을 경우, 응답에서 가장 먼저 등장하는 1~4 숫자를 fallback으로 사용
    fallback = re.search(r"[1-4]", response_text)
    return int(fallback.group(0)) if fallback else None


def main():
    if not API_KEY or API_KEY.strip() == "" or "여기에" in API_KEY:
        print("[오류] TIMELY_API_KEY가 설정되지 않았습니다. pytorch_port/.env를 확인해주세요.")
        sys.exit(1)

    questions = load_questions()
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)

    print(f"[info] 사용 모델: {MODEL}")
    print(f"[info] 총 {len(questions)}문항에 대해 AI 응답을 수집합니다...\n")

    results = []
    correct_count = 0

    for q in questions:
        image_path = os.path.join(IMAGE_DIR, f"q{q['id']:02d}.png")

        try:
            response_text = ask_model(image_path)
        except requests.exceptions.RequestException as e:
            print(f"  q{q['id']:02d}: [오류] 요청 실패 - {type(e).__name__}: {e}")
            results.append({
                "id": q["id"], "subject": q["subject"],
                "ai_response": None, "ai_answer_index": None,
                "correct_answer_index": q["answer_index"], "is_correct": False,
                "error": str(e),
            })
            continue

        ai_answer_number = parse_answer_number(response_text)
        ai_answer_index = ai_answer_number - 1 if ai_answer_number is not None else None
        is_correct = ai_answer_index == q["answer_index"]
        correct_count += int(is_correct)

        status = "O" if is_correct else "X"
        correct_choice = q["choices"][q["answer_index"]]
        ai_choice = q["choices"][ai_answer_index] if ai_answer_index is not None and 0 <= ai_answer_index < len(q["choices"]) else "파싱 실패"

        print(f"  q{q['id']:02d} ({q['subject']}) [{status}] AI 답: {ai_choice} / 정답: {correct_choice}")
        print(f"    원본 응답: {response_text.strip()}")

        results.append({
            "id": q["id"],
            "subject": q["subject"],
            "ai_response": response_text.strip(),
            "ai_answer_index": ai_answer_index,
            "correct_answer_index": q["answer_index"],
            "is_correct": is_correct,
        })

        time.sleep(0.5)  # API 호출 간 약간의 간격

    accuracy = correct_count / len(questions)
    summary = {
        "model": MODEL,
        "num_questions": len(questions),
        "correct_count": correct_count,
        "accuracy": round(accuracy, 4),
        "results": results,
    }

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[완료] 정답률: {correct_count}/{len(questions)} ({accuracy:.1%})")
    print(f"[결과 저장] {RESULT_PATH}")


if __name__ == "__main__":
    main()
