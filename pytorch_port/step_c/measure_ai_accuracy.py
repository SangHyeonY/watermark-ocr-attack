# -*- coding: utf-8 -*-
"""
Step C 결과 측정 (2/2): 공격된 이미지에 대한 이미지 입력 AI(Claude Sonnet 5) 정답률.

attack_questions.py가 만든 eps 3조건(low/mid/high)의 공격된 문제 이미지를
Claude Sonnet 5(Timely API)에게 보여주고, 실제로 정답을 맞히는지 측정한다.

Step B의 measure_ai_accuracy.py와 로직은 동일하되, 이미지 경로만
Step C의 조건별 결과 폴더를 가리키도록 바꿨다. Step B 베이스라인(공격 전,
정답률 100%)과 비교하기 위한 목적.

보안 주의:
- API 키는 .env 파일에서만 읽고, 코드에서 값을 출력하거나 로그에 남기지 않는다.

실행 방법:
    conda activate fawa
    python pytorch_port/step_c/measure_ai_accuracy.py
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

QUESTIONS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "step_b", "questions.json"
)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

CONDITIONS = ["low", "mid", "high"]  # attack_questions.py의 EPS_CONDITIONS와 동일

PROMPT_TEMPLATE = (
    "이 이미지는 4지선다 객관식 문제입니다. 문제를 읽고 정답을 골라주세요.\n"
    "반드시 아래 형식으로만 답하세요 (다른 설명 없이 알파벳만):\n"
    "정답: X\n"
    "(X는 A, B, C, D 중 하나이며, 선택지 순서대로 1번째=A), 2번째=B), 3번째=C), 4번째=D)입니다)"
)

LETTER_TO_NUMBER = {"A": 1, "B": 2, "C": 3, "D": 4}


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
    """AI 응답 텍스트에서 "정답: X" 형식의 알파벳(A~D)을 찾아 1~4 숫자로 변환한다.
    못 찾으면 None."""
    match = re.search(r"정답\s*[:：]?\s*([A-Da-d])", response_text)
    if match:
        return LETTER_TO_NUMBER[match.group(1).upper()]
    fallback = re.search(r"[A-Da-d]", response_text)
    return LETTER_TO_NUMBER[fallback.group(0).upper()] if fallback else None


def measure_condition(condition_name, questions):
    """한 eps 조건(low/mid/high)에 대해 20문항 전체의 AI 정답률을 측정한다."""
    image_dir = os.path.join(OUTPUT_DIR, f"question_images_{condition_name}")
    result_path = os.path.join(OUTPUT_DIR, f"question_images_{condition_name}", "ai_result.json")

    print(f"\n{'='*60}")
    print(f"[조건: {condition_name}]")
    print(f"{'='*60}")

    results = []
    correct_count = 0

    for q in questions:
        image_path = os.path.join(image_dir, f"q{q['id']:02d}.png")
        if not os.path.exists(image_path):
            print(f"  q{q['id']:02d}: [건너뜀] 이미지 없음 ({image_path})")
            continue

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
        ai_choice = (
            q["choices"][ai_answer_index]
            if ai_answer_index is not None and 0 <= ai_answer_index < len(q["choices"])
            else "파싱 실패"
        )

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

        time.sleep(0.5)

    accuracy = correct_count / len(results) if results else 0.0
    summary = {
        "condition": condition_name,
        "model": MODEL,
        "num_questions": len(results),
        "correct_count": correct_count,
        "accuracy": round(accuracy, 4),
        "results": results,
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[조건: {condition_name} 완료] 정답률: {correct_count}/{len(results)} ({accuracy:.1%})")
    print(f"[결과 저장] {result_path}")
    return summary


def main():
    if not API_KEY or API_KEY.strip() == "" or "여기에" in API_KEY:
        print("[오류] TIMELY_API_KEY가 설정되지 않았습니다. pytorch_port/.env를 확인해주세요.")
        sys.exit(1)

    questions = load_questions()
    print(f"[info] 사용 모델: {MODEL}")
    print(f"[info] 총 {len(questions)}문항 x {len(CONDITIONS)}개 조건에 대해 AI 응답을 수집합니다...")

    all_summaries = []
    for condition_name in CONDITIONS:
        summary = measure_condition(condition_name, questions)
        all_summaries.append(summary)

    print(f"\n{'='*60}")
    print("[전체 요약]")
    print(f"{'='*60}")
    print("  조건 (Step B 베이스라인: 20/20, 100%)")
    for s in all_summaries:
        print(f"  {s['condition']}: {s['correct_count']}/{s['num_questions']} ({s['accuracy']:.1%})")


if __name__ == "__main__":
    main()
