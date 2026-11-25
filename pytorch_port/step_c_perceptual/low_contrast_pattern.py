# -*- coding: utf-8 -*-
"""
Step C-Perceptual 파일럿 (2차): Low Contrast 기법.

배경 (Making MLLMs Blind, 2026, arXiv:2604.06950, Table 2 참고):
- "Low Contrast"는 SmuggleBench의 9개 교란 기법 중 GPT-5 대상 ASR(공격 성공률)이
  99.5%로 가장 높았던 기법이다. 원리는 단순하다: 글자색을 배경색과 아주 비슷하게
  만든다.
- 1차 파일럿(perceptual_pattern.py의 noise/jitter)과의 결정적 차이: 1차는
  "배경(워터마크 영역)"만 건드리고 글자는 그대로 뒀다 (EasyOCR 공격과 동일한
  설계 철학). 이 2차 파일럿은 반대로 "글자 자체"의 색을 바꾼다. 논문에서 효과가
  컸던 기법 대부분(Low Contrast, Tiny Text, Artistic 등)이 전부 "글자 자체"를
  건드리는 방식이었다는 점에서 착안했다.
- 중요한 트레이드오프: 연구계획서의 "사람 가독성 유지" 제약과 정면으로
  부딪힌다. 대비를 너무 낮추면 AI뿐 아니라 사람도 못 읽는다. 그래서 "AI는
  못 읽을 만큼, 사람은 읽을 수 있을 만큼"의 경계를 찾는 것이 이 파일럿의
  목적이다. 그 경계가 어디인지는 이론으로 알 수 없고, 실제로 여러 단계를
  만들어 사람이 눈으로, AI가 API로 각각 확인해봐야 한다.

이 모듈은 render_questions.py의 레이아웃 로직을 재사용하되, 글자색만
배경색(흰색)에 가깝게 바꾼 버전을 생성한다.
"""
import os
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "step_b"))
from render_questions import FONT_SIZE, LINE_SPACING, MARGIN, CHOICE_LABELS  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from watermark import resolve_font  # noqa: E402


def render_low_contrast_question(question_obj, gray_level, font_path=None,
                                   bg_color=(255, 255, 255)):
    """문제 이미지를 렌더링하되, 글자색을 지정한 회색조(gray_level)로 그린다.

    Args:
        question_obj: questions.json의 문항 1개 (dict)
        gray_level: 글자색의 밝기 (0=완전 검정, 255=배경과 동일한 흰색).
            값이 클수록 대비가 낮아져 사람도 AI도 읽기 어려워진다.
        font_path: 사용할 폰트 경로 (None이면 기본 한글 폰트)
        bg_color: 배경색 (기본값은 흰색, render_questions.py와 동일)

    Returns:
        PIL.Image (RGB)
    """
    font_path = resolve_font(font_path)
    font = ImageFont.truetype(font_path, FONT_SIZE)

    text_color = (gray_level, gray_level, gray_level)

    question_line = f"{question_obj['id']}. {question_obj['question']}"
    choice_lines = [
        f"{CHOICE_LABELS[i]} {choice}"
        for i, choice in enumerate(question_obj["choices"])
    ]
    all_lines = [question_line, ""] + choice_lines

    line_heights = []
    max_width = 0
    for line in all_lines:
        if line == "":
            line_heights.append(FONT_SIZE // 2)
            continue
        bbox = font.getbbox(line)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        max_width = max(max_width, w)
        line_heights.append(h)

    total_height = sum(line_heights) + LINE_SPACING * (len(all_lines) - 1) + 2 * MARGIN
    total_width = max_width + 2 * MARGIN

    img = Image.new("RGB", (total_width, total_height), color=bg_color)
    draw = ImageDraw.Draw(img)

    y = MARGIN
    for line, h in zip(all_lines, line_heights):
        if line != "":
            bbox = font.getbbox(line)
            draw.text((MARGIN - bbox[0], y - bbox[1]), line, fill=text_color, font=font)
        y += h + LINE_SPACING

    return img
