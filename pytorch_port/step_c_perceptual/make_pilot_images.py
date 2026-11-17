# -*- coding: utf-8 -*-
"""
Step C-Perceptual 파일럿: Step B 20문항 중 일부를 골라, 그래디언트 없이
설계한 지각적 패턴(perceptual_pattern.py) 4가지 조건으로 이미지를 생성한다.

조건:
    baseline - 원본 (패턴 없음, Step B와 동일)
    noise    - 고빈도 체커보드 노이즈만
    jitter   - 블록 지터만
    combined - 노이즈 + 지터 조합

실행 방법:
    conda activate fawa
    python pytorch_port/step_c_perceptual/make_pilot_images.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "step_b"))

from render_questions import load_questions as load_all_questions, render_question_image  # noqa: E402
from perceptual_pattern import high_freq_noise_pattern, jitter_distortion, combined_pattern  # noqa: E402

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# 파일럿이므로 20문항 전부가 아니라 일부만 사용 (시간/비용 절약)
# 다양성을 위해 과목이 다른 문항 5개를 선택: 한국사 2개, 과학/사회/상식 각 1개
PILOT_QUESTION_IDS = [1, 2, 6, 11, 16]

CONDITIONS = {
    "baseline": lambda img: {"composite": img, "wm_mask": None},
    "noise": lambda img: high_freq_noise_pattern(img, cell_size=3, intensity=40, seed=42),
    "jitter": lambda img: jitter_distortion(img, block_size=4, max_shift=3, seed=42),
    "combined": lambda img: combined_pattern(img, seed=42),
}


def main():
    all_questions = load_all_questions()
    questions_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "step_b", "questions.json"
    )
    id_to_q = {q["id"]: q for q in all_questions}
    pilot_questions = [id_to_q[qid] for qid in PILOT_QUESTION_IDS]

    pilot_labels = [f"q{q['id']:02d}({q['subject']})" for q in pilot_questions]
    print(f"[info] 파일럿 문항 {len(pilot_questions)}개 선택: {pilot_labels}")

    for condition_name, pattern_fn in CONDITIONS.items():
        cond_dir = os.path.join(OUTPUT_DIR, f"question_images_{condition_name}")
        os.makedirs(cond_dir, exist_ok=True)

        print(f"\n[조건: {condition_name}]")
        for q in pilot_questions:
            base_img = render_question_image(q)
            result = pattern_fn(base_img)
            save_path = os.path.join(cond_dir, f"q{q['id']:02d}.png")
            result["composite"].save(save_path)
            print(f"  q{q['id']:02d} ({q['subject']}): {result['composite'].size} -> {save_path}")

    # 파일럿에서 사용한 문항 목록을 저장해 measure 스크립트가 재사용할 수 있게 함
    pilot_questions_path = os.path.join(OUTPUT_DIR, "pilot_questions.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(pilot_questions_path, "w", encoding="utf-8") as f:
        json.dump(pilot_questions, f, ensure_ascii=False, indent=2)

    print(f"\n[완료] 파일럿 문항 목록 저장 -> {pilot_questions_path}")
    print(f"[완료] 조건별 이미지 생성 완료 -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
