# -*- coding: utf-8 -*-
"""
Step C 추가 실험: 토익(TOEIC) 스타일 문법 문제 5개에 워터마크 공격 적용.

배경:
- Step B/C에서 쓴 20문항(한국사/과학/사회/상식)은 "상식으로도 풀 수 있는" 쉬운
  문제라서, 이미지 입력 AI(Claude Sonnet 5)가 선택지 글자를 정확히 안 읽어도
  맞힐 수 있었다. 실제로 eps를 0.6까지 올려도 AI 정답률이 20/20(100%)에서
  전혀 안 떨어졌는데, 이게 "공격이 약해서"인지 "문제가 쉬워서"인지 구분이
  안 되는 상태였다.
- 그래서 "선택지를 정확히 봐야만 풀리는" 토익 문법 문제로 추가 검증을 한다.
  (실제 AI 글래스를 이용한 시험 부정행위가 적발된 사례가 토익이었다는 점도
  고려한 선택 - WORK_LOG_StepC.md 참고)

사용하는 공격 파이프라인은 attack_questions.py의 run_attack_pipeline()을
그대로 재사용한다 (문항 리스트와 결과 폴더만 다름).

실행 방법:
    conda activate fawa
    python pytorch_port/step_c/attack_toeic.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from attack_questions import run_attack_pipeline  # noqa: E402

QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "toeic_questions.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs_toeic")


def load_toeic_questions():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    questions = load_toeic_questions()
    print(f"[info] 토익 문항 {len(questions)}개로 공격 파이프라인을 실행합니다...")
    run_attack_pipeline(questions, OUTPUT_DIR)


if __name__ == "__main__":
    main()
