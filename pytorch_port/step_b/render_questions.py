# -*- coding: utf-8 -*-
"""
Step B 20문항(questions.json)을 "문제 + 4개 선택지"가 포함된 하나의 이미지로
렌더링하는 스크립트.

기존 watermark.py의 render_text_image()는 "한 단어/한 줄"만 그릴 수 있어서,
여러 줄(문제 1줄 + 선택지 4줄)로 구성된 객관식 문제 이미지를 만들려면
별도 함수가 필요하다. 이 파일은 그 역할만 담당한다 (워터마크 합성은 아직
하지 않음 - Step B는 "보호 처리 전 원본" 베이스라인 측정 단계이기 때문).

실행 방법:
    conda activate fawa
    python pytorch_port/step_b/render_questions.py
"""
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from watermark import resolve_font  # noqa: E402

QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "questions.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "question_images")

FONT_SIZE = 32
LINE_SPACING = 16
MARGIN = 30
CHOICE_LABELS = ["①", "②", "③", "④"]


def load_questions():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def render_question_image(question_obj, font_path=None):
    """문제 1개를 "문제 + 4개 선택지"가 포함된 하나의 이미지로 렌더링한다.

    레이아웃:
        1. 문항 번호 + 질문 텍스트 (1줄)
        2. 빈 줄
        3. 선택지 4개 (①~④, 각 1줄씩)
    """
    font_path = resolve_font(font_path)
    font = ImageFont.truetype(font_path, FONT_SIZE)

    question_line = f"{question_obj['id']}. {question_obj['question']}"
    choice_lines = [
        f"{CHOICE_LABELS[i]} {choice}"
        for i, choice in enumerate(question_obj["choices"])
    ]
    all_lines = [question_line, ""] + choice_lines

    # 각 줄의 높이를 측정해서 전체 이미지 크기를 계산
    line_heights = []
    max_width = 0
    for line in all_lines:
        if line == "":
            line_heights.append(FONT_SIZE // 2)  # 빈 줄은 절반 높이만 사용
            continue
        bbox = font.getbbox(line)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        max_width = max(max_width, w)
        line_heights.append(h)

    total_height = sum(line_heights) + LINE_SPACING * (len(all_lines) - 1) + 2 * MARGIN
    total_width = max_width + 2 * MARGIN

    img = Image.new("RGB", (total_width, total_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    y = MARGIN
    for line, h in zip(all_lines, line_heights):
        if line != "":
            bbox = font.getbbox(line)
            draw.text((MARGIN - bbox[0], y - bbox[1]), line, fill=(0, 0, 0), font=font)
        y += h + LINE_SPACING

    return img


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    questions = load_questions()

    print(f"[info] {len(questions)}개 문항을 이미지로 렌더링합니다...")
    for q in questions:
        img = render_question_image(q)
        save_path = os.path.join(OUTPUT_DIR, f"q{q['id']:02d}.png")
        img.save(save_path)
        print(f"  q{q['id']:02d} ({q['subject']}): {img.size} -> {save_path}")

    print(f"\n[완료] 총 {len(questions)}개 문항 이미지 생성 완료 -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
