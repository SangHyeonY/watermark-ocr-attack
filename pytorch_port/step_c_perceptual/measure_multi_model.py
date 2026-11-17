# -*- coding: utf-8 -*-
"""
Step C-Perceptual 파일럿 (3차): Low Contrast 패턴이 Claude 외 다른 모델
(GPT-4o, Gemini 2.5 Pro)에도 안 통하는지, 아니면 Claude만 유독 강건한지 비교.

배경:
- 2차 파일럿(measure_ai_accuracy.py)에서 Low Contrast(글자색 235/255까지)가
  Claude Sonnet 5에는 전혀 안 통했다 (4단계 전부 5/5 정답).
- SmuggleBench 논문(Making MLLMs Blind, 2026)은 GPT-5 기준 Low Contrast ASR
  99.5%를 보고했는데, Claude는 테스트 대상에 없었다.
- "Claude가 유독 강건한 것"인지 "우리 테스트 설계(단순 흰 배경) 자체가
  문제"인지 구분하기 위해, 같은 이미지를 GPT-4o/Gemini에도 보여준다.
  - 다른 모델도 다 안 통하면: 테스트 설계(이미지가 너무 단순함, API 전처리
    가능성 등) 문제일 가능성이 높아진다.
  - 다른 모델은 통하는데 Claude만 특이하게 버티면: Claude의 강건성 자체가
    의미 있는 발견이 된다.

실행 방법:
    conda activate fawa
    python pytorch_port/step_c_perceptual/measure_multi_model.py
"""
import base64
import json
import os
import re
import sys
import time

import requests
from dotenv import load_dotenv

ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(ENV_PATH)

API_KEY = os.environ.get("TIMELY_API_KEY")
BASE_URL = os.environ.get("TIMELY_BASE_URL", "https://hello.timelygpt.co.kr/api/v2/chat/bridge/openai")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
PILOT_QUESTIONS_PATH = os.path.join(OUTPUT_DIR, "pilot_questions.json")

# 비용 절약을 위해 가장 극단적인 조건(235, 거의 안 보임)과
# 베이스라인(대비 최대)만 비교한다 (4단계 전부 X, 2단계만)
CONDITIONS = ["contrast_full", "contrast_235"]
MODELS = ["anthropic/claude-sonnet-5", "openai/gpt-4o", "google/gemini-2.5-pro"]

PROMPT_TEMPLATE = (
    "이 이미지는 4지선다 객관식 문제입니다. 문제를 읽고 정답을 골라주세요.\n"
    "반드시 아래 형식으로만 답하세요 (다른 설명 없이 알파벳만):\n"
    "정답: X\n"
    "(X는 A, B, C, D 중 하나이며, 선택지 순서대로 1번째=A), 2번째=B), 3번째=C), 4번째=D)입니다)"
)

LETTER_TO_NUMBER = {"A": 1, "B": 2, "C": 3, "D": 4}


def load_pilot_questions():
    with open(PILOT_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def ask_model(image_path, model):
    image_b64 = encode_image_to_base64(image_path)
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT_TEMPLATE},
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


def parse_answer_number(response_text):
    match = re.search(r"정답\s*[:：]?\s*([A-Da-d])", response_text)
    if match:
        return LETTER_TO_NUMBER[match.group(1).upper()]
    fallback = re.search(r"[A-Da-d]", response_text)
    return LETTER_TO_NUMBER[fallback.group(0).upper()] if fallback else None


def measure(model, condition_name, questions):
    image_dir = os.path.join(OUTPUT_DIR, f"question_images_{condition_name}")
    results = []
    correct_count = 0

    for q in questions:
        image_path = os.path.join(image_dir, f"q{q['id']:02d}.png")
        if not os.path.exists(image_path):
            continue
        try:
            response_text = ask_model(image_path, model)
        except Exception as e:
            print(f"    q{q['id']:02d}: [오류] {type(e).__name__}: {e}")
            results.append({"id": q["id"], "is_correct": False, "error": str(e)})
            continue

        ai_answer_number = parse_answer_number(response_text)
        ai_answer_index = ai_answer_number - 1 if ai_answer_number is not None else None
        is_correct = ai_answer_index == q["answer_index"]
        correct_count += int(is_correct)

        status = "O" if is_correct else "X"
        print(f"    q{q['id']:02d} [{status}] 응답: {response_text.strip()[:60]}")

        results.append({
            "id": q["id"], "is_correct": is_correct,
            "ai_response": response_text.strip(),
        })
        time.sleep(0.5)

    accuracy = correct_count / len(results) if results else 0.0
    return {"correct_count": correct_count, "num_questions": len(results),
            "accuracy": accuracy, "results": results}


def main():
    if not API_KEY or API_KEY.strip() == "" or "여기에" in API_KEY:
        print("[오류] TIMELY_API_KEY가 설정되지 않았습니다.")
        sys.exit(1)

    questions = load_pilot_questions()
    print(f"[info] 파일럿 문항 {len(questions)}개 x 모델 {len(MODELS)}개 x 조건 {len(CONDITIONS)}개")
    print(f"[info] 모델: {MODELS}")
    print(f"[info] 조건: {CONDITIONS}")

    all_summaries = {}
    for model in MODELS:
        all_summaries[model] = {}
        print(f"\n{'='*70}\n[모델: {model}]\n{'='*70}")
        for condition_name in CONDITIONS:
            print(f"\n  [조건: {condition_name}]")
            summary = measure(model, condition_name, questions)
            all_summaries[model][condition_name] = summary
            print(f"  -> 정답률: {summary['correct_count']}/{summary['num_questions']} "
                  f"({summary['accuracy']:.1%})")

    result_path = os.path.join(OUTPUT_DIR, "multi_model_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}\n[전체 요약: 모델 x 조건별 정답률]\n{'='*70}")
    header = "모델".ljust(28) + "".join(c.ljust(18) for c in CONDITIONS)
    print(header)
    for model in MODELS:
        row = model.ljust(28)
        for condition_name in CONDITIONS:
            s = all_summaries[model][condition_name]
            row += f"{s['correct_count']}/{s['num_questions']} ({s['accuracy']:.0%})".ljust(18)
        print(row)

    print(f"\n[결과 저장] {result_path}")


if __name__ == "__main__":
    main()
