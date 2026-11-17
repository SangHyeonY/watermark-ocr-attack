# -*- coding: utf-8 -*-
"""
토익 5문항 베이스라인 + Step C 공격 후, 이미지 입력 AI(Claude Sonnet 5) 정답률 측정.

두 가지를 측정한다:
1. 베이스라인: 워터마크/공격 없는 원본 이미지 (이 스크립트에서 직접 렌더링)
2. 공격 후: attack_toeic.py가 만든 eps 3조건(low/mid/high) 결과 이미지
   (outputs_toeic/question_images_{condition}/ 폴더)

실행 방법:
    conda activate fawa
    python pytorch_port/step_c/measure_ai_accuracy_toeic.py
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

QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "toeic_questions.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs_toeic")
CONDITIONS = ["low", "mid", "high"]

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "step_b"))
from render_questions import render_question_image  # noqa: E402

PROMPT_TEMPLATE = (
    "이 이미지는 4지선다 영어 문법 문제입니다. 문제를 읽고 정답을 골라주세요.\n"
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


def ask_model_with_pil(pil_image):
    """PIL 이미지 객체를 직접 받아 base64로 인코딩 후 API에 전송 (베이스라인용,
    파일로 먼저 저장하지 않고 바로 전송하기 위함)."""
    import io
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return _send_request(image_b64)


def ask_model_with_path(image_path):
    image_b64 = encode_image_to_base64(image_path)
    return _send_request(image_b64)


def _send_request(image_b64):
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
    match = re.search(r"정답\s*[:：]?\s*([A-Da-d])", response_text)
    if match:
        return LETTER_TO_NUMBER[match.group(1).upper()]
    fallback = re.search(r"[A-Da-d]", response_text)
    return LETTER_TO_NUMBER[fallback.group(0).upper()] if fallback else None


def evaluate(questions, get_response_fn, label):
    print(f"\n{'='*60}")
    print(f"[{label}]")
    print(f"{'='*60}")

    results = []
    correct_count = 0

    for q in questions:
        try:
            response_text = get_response_fn(q)
        except requests.exceptions.RequestException as e:
            print(f"  q{q['id']}: [오류] {type(e).__name__}: {e}")
            results.append({
                "id": q["id"], "ai_response": None, "is_correct": False, "error": str(e),
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
        print(f"  q{q['id']} [{status}] AI 답: {ai_choice} / 정답: {correct_choice} (응답: {response_text.strip()})")

        results.append({
            "id": q["id"],
            "ai_response": response_text.strip(),
            "ai_answer_index": ai_answer_index,
            "correct_answer_index": q["answer_index"],
            "is_correct": is_correct,
        })
        time.sleep(0.5)

    accuracy = correct_count / len(results) if results else 0.0
    print(f"[{label} 결과] {correct_count}/{len(results)} ({accuracy:.1%})")
    return {"label": label, "num_questions": len(results), "correct_count": correct_count,
            "accuracy": round(accuracy, 4), "results": results}


def main():
    if not API_KEY or API_KEY.strip() == "" or "여기에" in API_KEY:
        print("[오류] TIMELY_API_KEY가 설정되지 않았습니다.")
        sys.exit(1)

    questions = load_questions()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_summaries = []

    # 1) 베이스라인 (공격 없는 원본, 직접 렌더링)
    def get_baseline_response(q):
        img = render_question_image(q)
        return ask_model_with_pil(img)

    baseline_summary = evaluate(questions, get_baseline_response, "베이스라인 (공격 전)")
    all_summaries.append(baseline_summary)
    with open(os.path.join(OUTPUT_DIR, "ai_baseline_result.json"), "w", encoding="utf-8") as f:
        json.dump(baseline_summary, f, ensure_ascii=False, indent=2)

    # 2) eps 3조건 (공격 후, attack_toeic.py가 만든 이미지 사용)
    for condition in CONDITIONS:
        image_dir = os.path.join(OUTPUT_DIR, f"question_images_{condition}")

        def get_condition_response(q, image_dir=image_dir):
            path = os.path.join(image_dir, f"q{q['id']:02d}.png")
            return ask_model_with_path(path)

        summary = evaluate(questions, get_condition_response, f"조건: {condition}")
        all_summaries.append(summary)
        with open(os.path.join(image_dir, "ai_result.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print("[전체 요약]")
    print(f"{'='*60}")
    for s in all_summaries:
        print(f"  {s['label']}: {s['correct_count']}/{s['num_questions']} ({s['accuracy']:.1%})")


if __name__ == "__main__":
    main()
