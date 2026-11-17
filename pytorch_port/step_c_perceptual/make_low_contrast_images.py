# -*- coding: utf-8 -*-
"""
Step C-Perceptual 파일럿 (2차): Low Contrast 단계별 이미지 생성.

배경색(255,255,255, 흰색)에 대해 글자색(gray_level)을 여러 단계로 조절해
"AI는 못 읽는데 사람은 읽을 수 있는" 경계가 어디인지 탐색한다.

단계 선정 기준:
- gray_level 0 (완전 검정) = 기존 Step B/C 베이스라인과 동일 (대비 최대)
- gray_level 255 (완전 흰색) = 글자가 완전히 사라짐 (대비 0, 당연히 둘 다 못 읽음)
- 그 사이 값들을 촘촘히 배치해 "AI가 못 읽기 시작하는 지점"과
  "사람이 못 읽기 시작하는 지점"의 차이(간극)를 찾는다.

실행 방법:
    conda activate fawa
    python pytorch_port/step_c_perceptual/make_low_contrast_images.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "step_b"))

from render_questions import load_questions as load_all_questions  # noqa: E402
from low_contrast_pattern import render_low_contrast_question  # noqa: E402

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# 기존 노이즈/지터 파일럿과 동일한 5문항 사용 (일관성 유지, 결과 비교 용이)
PILOT_QUESTION_IDS = [1, 2, 6, 11, 16]

# gray_level별 조건 이름. 숫자가 클수록 대비가 낮음(배경 흰색에 더 가까움).
CONTRAST_LEVELS = {
    "contrast_full": 0,      # 기존 베이스라인과 동일 (완전 검정 글자)
    "contrast_200": 200,     # 논문 수준에 가까운 저대비 (옅은 회색)
    "contrast_220": 220,     # 더 낮은 대비
    "contrast_235": 235,     # 거의 안 보이는 수준 (SmuggleBench Low Contrast에 가까움)
}


def main():
    all_questions = load_all_questions()
    id_to_q = {q["id"]: q for q in all_questions}
    pilot_questions = [id_to_q[qid] for qid in PILOT_QUESTION_IDS]

    pilot_labels = [f"q{q['id']:02d}({q['subject']})" for q in pilot_questions]
    print(f"[info] 파일럿 문항 {len(pilot_questions)}개 선택: {pilot_labels}")

    for condition_name, gray_level in CONTRAST_LEVELS.items():
        cond_dir = os.path.join(OUTPUT_DIR, f"question_images_{condition_name}")
        os.makedirs(cond_dir, exist_ok=True)

        print(f"\n[조건: {condition_name}] (글자색 회색조 = {gray_level}/255, 배경 = 255)")
        for q in pilot_questions:
            img = render_low_contrast_question(q, gray_level=gray_level)
            save_path = os.path.join(cond_dir, f"q{q['id']:02d}.png")
            img.save(save_path)
            print(f"  q{q['id']:02d} ({q['subject']}): {img.size} -> {save_path}")

    pilot_questions_path = os.path.join(OUTPUT_DIR, "pilot_questions.json")
    with open(pilot_questions_path, "w", encoding="utf-8") as f:
        json.dump(pilot_questions, f, ensure_ascii=False, indent=2)

    print(f"\n[완료] 조건별 이미지 생성 완료 -> {OUTPUT_DIR}")
    print(f"[완료] 조건 목록: {list(CONTRAST_LEVELS.keys())}")


if __name__ == "__main__":
    main()
