# -*- coding: utf-8 -*-
"""
텍스트 비교 지표(편집 거리, CER) 계산 유틸리티.

여러 곳(Step B의 OCR 베이스라인 측정, Step C의 공격 성공 판정)에서 똑같은
"두 문자열이 얼마나 다른지"를 재는 로직이 필요해서, 중복을 피하려고 공통
모듈로 분리했다. (원래는 step_b/measure_ocr_baseline.py에만 있었음)
"""


def levenshtein_distance(s1, s2):
    """두 문자열 간 편집 거리(Levenshtein distance)를 계산한다.
    편집 거리 = 한 문자열을 다른 문자열로 바꾸는 데 필요한 최소
    삽입/삭제/대체 횟수."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def compute_cer(hypothesis, reference):
    """CER(Character Error Rate, 문자 단위 오류율)을 계산한다.

    CER = levenshtein_distance(hypothesis, reference) / len(reference)
    값이 0에 가까울수록 hypothesis가 reference와 가깝다(=정확하게 인식됨).
    """
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    distance = levenshtein_distance(hypothesis, reference)
    return distance / len(reference)
