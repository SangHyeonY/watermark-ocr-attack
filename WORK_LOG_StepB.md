# 작업 로그 — Step B: 20문항 베이스라인 측정

> 작업정리 문서의 **Step B: 20문항 예비실험 베이스라인 측정**을 실행한 기록.
> 전체 진행 상황은 `WORK_LOG_INDEX.md` 참고.

## 한 일

1. 20문항 객관식(4지선다) 문제 제작 — 한국사/과학/사회/상식 혼합, 쉬운 난이도
   (`pytorch_port/step_b/questions.json`)
2. 각 문항을 "문제 + 선택지 4개"가 담긴 이미지로 렌더링
   (`pytorch_port/step_b/render_questions.py`)
3. 보호 패턴을 적용하지 않은 원본 이미지로 베이스라인 두 가지 측정:
   - OCR(EasyOCR) 문자 인식 정확도 (`measure_ocr_baseline.py`)
   - 이미지 입력 AI(Claude Sonnet 5, Timely API) 정답률 (`measure_ai_accuracy.py`)

## 결과

| 측정 항목 | 결과 |
|---|---|
| OCR 평균 CER(문자 오류율) | 17.99% |
| AI 정답률 | 20/20 (100%) |

## 발견한 점

- OCR 오류율이 꽤 높게 나왔지만, 원인을 보니 대부분 **보호 패턴과 무관한 것**이었다.
  - 원형 숫자(①②③④)를 EasyOCR이 전혀 인식하지 못함
  - 비슷한 숫자를 헷갈림 (예: "11." → "77.")
  - 한글 받침 인식 오류 (예: "건국을" → "건국올")
- 반면 AI는 이런 OCR 오류와 무관하게 **정답률 100%**를 기록함.
  → "OCR 오류가 늘어도 AI 정답률은 그대로일 수 있다"는 걸 베이스라인 단계에서 확인.
  이후 Step C에서 워터마크를 적용했을 때 OCR 오류율과 AI 정답률이 각각 어떻게
  변하는지 비교할 때, 이 둘을 구분해서 봐야 한다는 근거가 됨.

## 생성된 파일

- `pytorch_port/step_b/questions.json` — 20문항 데이터
- `pytorch_port/step_b/render_questions.py` — 문항 이미지 렌더링
- `pytorch_port/step_b/measure_ocr_baseline.py` — OCR 베이스라인 측정
- `pytorch_port/step_b/measure_ai_accuracy.py` — AI 정답률 측정
- `pytorch_port/step_b/outputs/` — 결과물(문항 이미지, 측정 결과 JSON)

## 다음 단계

Step C — 워터마크 패턴을 20문항에 적용하고, 위 베이스라인과 비교해 AI 성능/사람
가독성이 어떻게 달라지는지 측정.
