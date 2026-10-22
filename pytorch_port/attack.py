# -*- coding: utf-8 -*-
"""
FAWA(Fast Adversarial Watermark Attack) 공격 루프의 PyTorch 포팅.

원본 참고: wm_grad.py의 "run attack" 섹션 (Gradient 기반, 논문 제목의 "Fast" 버전)

공격의 핵심 아이디어 (쉬운 설명):
- 우리는 "이 이미지가 틀린 글자(target_text)로 인식되게 만드는 손실(CTC loss)"을
  계산할 수 있고, 이 손실을 역전파(backward)하면 "입력 이미지의 각 픽셀을 어느
  방향으로 바꾸면 그 손실이 줄어드는지(=더 틀리게 만들 수 있는지)"를 알 수 있다.
  이 방향 정보가 바로 gradient(기울기)다.
- 이 gradient를 보고 픽셀을 조금씩 수정하는데, 한 번에 다 바꾸면 사람 눈에도
  보일 정도로 이미지가 망가질 수 있으므로, 아주 작은 step(eps_iter)만큼씩
  여러 번(iteration) 반복해서 서서히 바꾸고, 전체 변화량도 eps 이하로 제한한다.
- 가장 중요한 제약: 이 변화를 "워터마크 영역(wm_mask)"에만 적용한다.
  실제 글자가 있는 영역(text_mask)은 전혀 건드리지 않으므로, 사람이 읽는 글자는
  원본과 완전히 동일하게 유지된다. 즉 "배경 워터마크 무늬만 미세하게 바뀌어서
  AI를 속이는" 방식이다.

원본과의 차이점:
- 원본(TF1, wm_grad.py)은 L1/L2/Linf 세 가지 섭동 방식을 모두 구현했지만,
  실제로 쉘 스크립트에서 쓰인 것은 L2(pert_type='2')였다. 여기서는 구조를
  단순화하기 위해 Linf(sign 기반) 방식만 구현한다. 이는 FGSM/PGD 계열의
  표준적인 방식이며, 논문 제목의 "Fast"에 해당하는 접근이다.
- 모멘텀(여러 iteration에 걸친 gradient 누적)은 원본과 동일하게 유지한다.
"""
import numpy as np
import torch


def attack_watermark_region(
    ocr_wrapper,
    original_tensor,
    wm_mask_tensor,
    target_text,
    eps=0.2,
    eps_iter=0.02,
    max_iter=200,
    verbose=True,
):
    """워터마크 영역(wm_mask)에만 적대적 섭동을 가해 OCR을 속이는 공격을 수행한다.

    Args:
        ocr_wrapper: OcrAttackModel 인스턴스 (ocr_model.py)
        original_tensor: torch.Tensor, shape (1, 1, H, W), [-1, 1] 범위.
            attack 대상이 되는 원본 이미지 텐서 (OcrAttackModel.image_to_tensor로 생성).
        wm_mask_tensor: torch.Tensor, shape (1, 1, H, W), 값은 0.0 또는 1.0.
            워터마크가 있는 픽셀 위치만 1.0인 마스크. 이 마스크 밖의 영역은
            공격 중에도 절대 바뀌지 않는다 (= 실제 글자는 그대로 유지됨).
        target_text: 공격 목표 문자열 (원본과 다른 문자열이어야 공격 의미가 있음).
        eps: 전체 누적 변화량의 최대 한도 ([-1,1] 스케일 기준, 원본 기본값 0.2).
        eps_iter: 한 번의 iteration에서 움직이는 step 크기.
        max_iter: 최대 반복 횟수.
        verbose: True면 진행 상황을 중간중간 출력.

    Returns:
        dict with keys:
            "adv_tensor": 최종 적대적 이미지 텐서 (성공/실패 여부와 무관하게 마지막 상태)
            "success": bool, target_text와 완전히 같아졌는지 여부
            "success_iter": 성공한 iteration 번호 (실패 시 None)
            "final_prediction": 마지막 iteration에서의 OCR 예측 문자열
            "history": 각 iteration의 (prediction, ctc_loss) 리스트 (디버깅/그래프용)
    """
    clip_min, clip_max = -1.0, 1.0  # OcrAttackModel.image_to_tensor의 출력 범위

    adv_tensor = original_tensor.clone().detach()
    momentum = torch.zeros_like(adv_tensor)

    success = False
    success_iter = None
    history = []

    for it in range(max_iter):
        adv_tensor = adv_tensor.clone().detach().requires_grad_(True)

        # CTC loss: 이 값이 작을수록 "target_text로 읽힐 확률이 높음" = 공격에 가까움
        loss = ocr_wrapper.ctc_loss(adv_tensor, target_text=target_text)
        loss.backward()

        grad = adv_tensor.grad.detach()

        # gradient 정규화 (원본: tf.reduce_mean(tf.abs(grad))로 나눔)
        # -> iteration마다 gradient 크기가 들쭉날쭉한 것을 완화해서 step 크기를 일정하게 유지
        avoid_zero_div = 1e-12
        divisor = grad.abs().mean(dim=list(range(1, grad.dim())), keepdim=True)
        norm_grad = grad / torch.clamp(divisor, min=avoid_zero_div)

        # 모멘텀 누적 (원본: acc_m = m + norm_grad)
        momentum = momentum + norm_grad

        # 워터마크 영역에만 섭동이 적용되도록 마스크 곱
        masked_grad = momentum * wm_mask_tensor

        # loss를 줄이는 방향 = -gradient 방향. sign()으로 방향만 취해 일정한 step 이동.
        step = -eps_iter * torch.sign(masked_grad)

        with torch.no_grad():
            adv_tensor = adv_tensor.detach() + step
            # 원본 이미지 대비 전체 누적 변화량을 eps로 제한
            delta = torch.clamp(adv_tensor - original_tensor, min=-eps, max=eps)
            adv_tensor = original_tensor + delta
            # 픽셀 값 자체도 유효 범위로 제한
            adv_tensor = torch.clamp(adv_tensor, clip_min, clip_max)

        # 현재 상태에서 OCR이 뭐라고 읽는지 확인
        prediction = ocr_wrapper.predict_text(adv_tensor)
        history.append((prediction, loss.item()))

        if verbose and (it % 20 == 0 or it == max_iter - 1):
            print(f"  [iter {it:3d}] CTC loss={loss.item():.4f}, 현재 예측='{prediction}'")

        if prediction == target_text and not success:
            success = True
            success_iter = it
            if verbose:
                print(f"  -> 공격 성공! iteration {it}에서 목표 문자열 '{target_text}'로 바뀜")
            break

    return {
        "adv_tensor": adv_tensor.detach(),
        "success": success,
        "success_iter": success_iter,
        "final_prediction": history[-1][0] if history else None,
        "history": history,
    }


def attack_untargeted(
    ocr_wrapper,
    original_tensor,
    wm_mask_tensor,
    original_text,
    ground_truth_text=None,
    eps=0.45,
    eps_iter=0.045,
    max_iter=200,
    verbose=True,
):
    """워터마크 영역에만 적대적 섭동을 가해, "원래 글자로 읽히지 않게만" 만드는 공격.

    targeted 공격(attack_watermark_region)과의 차이 (쉬운 설명):
    - targeted: "엉뚱한 특정 글자(예: 가나다)로 읽게 만들자" - 매우 구체적인 목표라
      어려움. 실험해보니 eps를 크게 키워야 가끔 성공했다 (WORK_LOG.md 5단계 참고).
    - untargeted: "정답 글자(원본 그대로)로만 안 읽히면 성공" - 목표가 훨씬
      느슨해서 공격이 더 쉽고, 실제 연구 목적("AI가 문제를 못 읽게 방해하면 충분
      하다")과도 더 맞다.

    원리:
    - "원본 글자(original_text)로 정확히 읽히게 하는 손실(CTC loss)"을 계산한다.
      이 손실이 작을수록 "정답으로 잘 읽힌다"는 뜻이다.
    - targeted 공격은 이 손실을 줄이는 방향(gradient descent)으로 움직였지만,
      untargeted 공격은 반대로 이 손실을 키우는 방향(gradient ascent)으로
      움직인다. 즉 "정답을 맞히기 어렵게" 만드는 방향이다.
    - sign(grad)의 + 방향으로 이동하면 손실이 커지므로(ascent), targeted와
      달리 step에 음수를 붙이지 않는다.

    성공 판정 기준의 변천 (중요, WORK_LOG_StepC.md 참고):
    1차: "원본 글자(original_text)와 다르게 읽히면 성공". 짧은 단어(Step A)에서는
      통했지만, 문장/선택지(Step C)에서는 공격 전부터 이미 틀리게 읽는 경우가
      많아 의미가 없었다.
    2차: "공격 전(0번째 반복) OCR 예측과 다르면 성공"으로 수정. 그런데 실제로
      성공으로 집계된 사례를 들여다보니, 세 가지 전혀 다른 경우가 섞여 있었다:
      (a) 진짜로 더 틀려진 경우(예: "C) 이방원"->"C 이방원", 닫는 괄호 소실),
      (b) 오히려 더 정확해진 경우(예: "C) 베이장"->"C) 베이징", 우연히 오타가
      고쳐짐), (c) 의미 없는 변화(예: "B) 왕건"->"B) 왕건 ", 끝에 공백만 추가).
      (b), (c)는 공격 효과라고 볼 수 없는데 전부 "성공 1건"으로 잘못 집계됐다.
    3차(현재): "정답(ground_truth_text)과의 CER(문자 오류율)이 공격 전보다
      공격 후에 실제로 늘어났는가"로 최종 변경. 이러면 (a)만 성공으로 잡히고
      (b), (c)는 자동으로 걸러진다. 또한 반복 중 CER이 가장 높아진(가장 많이
      틀어진) 시점을 "최선의 공격 결과"로 추적한다 (중간에 우연히 한 번
      나빠졌다가 다시 좋아지는 경우를 대비해, 매 iteration의 최댓값을 계속
      추적하는 방식).

    Args:
        ocr_wrapper, original_tensor, wm_mask_tensor: attack_watermark_region과 동일.
        original_text: CTC loss 계산의 기준 텍스트 (정답으로부터 멀어지는 방향으로
            공격). 보통 공격 전 OCR 예측 결과를 그대로 넘기면 된다.
        ground_truth_text: 이 이미지의 실제 정답 텍스트. CER 계산 기준으로 쓰인다.
            None이면 original_text를 정답으로 간주한다 (Step A처럼 공격 전
            OCR이 항상 정답을 맞히는 짧은 단어에 해당).
        eps, eps_iter, max_iter, verbose: attack_watermark_region과 동일.
            eps 기본값(0.45)은 5개 한글 단어로 0.3~0.6 구간을 비교해서 정한 값으로,
            "공격 성공률 100%를 유지하는 가장 작은 값"이다.

    Returns:
        dict with keys:
            "adv_tensor": CER이 가장 높았던(가장 많이 틀어진) 시점의 이미지 텐서
                (실패 시에는 마지막 iteration의 텐서)
            "success": bool, 최선의 CER이 베이스라인 CER보다 실제로 높아졌는지 여부
            "success_iter": 그 최선의 CER이 나온 iteration 번호 (실패 시 None)
            "baseline_prediction": 공격을 시작하기 전(0번째 반복) OCR 예측 결과
            "baseline_cer": 공격 전 CER (ground_truth_text 대비)
            "final_prediction": 최선의 공격 결과에서의 OCR 예측 문자열
            "final_cer": 최선의 공격 결과에서의 CER
            "history": 각 iteration의 (prediction, ctc_loss, cer) 리스트
    """
    from text_metrics import compute_cer  # 순환 임포트 방지를 위해 함수 내부에서 임포트

    clip_min, clip_max = -1.0, 1.0

    if ground_truth_text is None:
        ground_truth_text = original_text

    adv_tensor = original_tensor.clone().detach()
    momentum = torch.zeros_like(adv_tensor)

    # 공격을 시작하기 전, 원본(워터마크 합성까지만 된 상태) 이미지를 OCR이
    # 어떻게 읽는지 베이스라인으로 기록한다.
    baseline_prediction = ocr_wrapper.predict_text(adv_tensor)
    baseline_cer = compute_cer(baseline_prediction.strip(), ground_truth_text.strip())

    # "지금까지 CER이 가장 높았던(가장 많이 틀어진) 시점"을 추적한다.
    best_cer = baseline_cer
    best_prediction = baseline_prediction
    best_tensor = adv_tensor.clone().detach()
    best_iter = None

    history = []

    for it in range(max_iter):
        adv_tensor = adv_tensor.clone().detach().requires_grad_(True)

        # "정답으로 읽히게 하는" 손실. 이 값이 클수록 "정답을 못 맞히고 있다"는 뜻.
        loss = ocr_wrapper.ctc_loss(adv_tensor, target_text=original_text)
        loss.backward()

        grad = adv_tensor.grad.detach()

        avoid_zero_div = 1e-12
        divisor = grad.abs().mean(dim=list(range(1, grad.dim())), keepdim=True)
        norm_grad = grad / torch.clamp(divisor, min=avoid_zero_div)

        momentum = momentum + norm_grad
        masked_grad = momentum * wm_mask_tensor

        # loss를 키우는(ascent) 방향 = +gradient 방향 (targeted와 부호가 반대)
        step = eps_iter * torch.sign(masked_grad)

        with torch.no_grad():
            adv_tensor = adv_tensor.detach() + step
            delta = torch.clamp(adv_tensor - original_tensor, min=-eps, max=eps)
            adv_tensor = original_tensor + delta
            adv_tensor = torch.clamp(adv_tensor, clip_min, clip_max)

        prediction = ocr_wrapper.predict_text(adv_tensor)
        cer = compute_cer(prediction.strip(), ground_truth_text.strip())
        history.append((prediction, loss.item(), cer))

        if cer > best_cer:
            best_cer = cer
            best_prediction = prediction
            best_tensor = adv_tensor.clone().detach()
            best_iter = it

        if verbose and (it % 20 == 0 or it == max_iter - 1):
            print(
                f"  [iter {it:3d}] CTC loss(정답 기준)={loss.item():.4f}, "
                f"현재 예측='{prediction}', CER={cer:.3f} (베이스라인 CER={baseline_cer:.3f})"
            )

    success = best_iter is not None  # best_cer > baseline_cer인 경우에만 best_iter가 채워짐
    if verbose:
        if success:
            print(
                f"  -> 공격 성공! iteration {best_iter}에서 CER이 {baseline_cer:.3f} -> "
                f"{best_cer:.3f}로 증가 (예측: '{baseline_prediction}' -> '{best_prediction}')"
            )
        else:
            print(f"  -> 공격 실패. CER이 베이스라인({baseline_cer:.3f}) 이상으로 오르지 않음.")

    return {
        "adv_tensor": best_tensor,
        "success": success,
        "success_iter": best_iter,
        "baseline_prediction": baseline_prediction,
        "baseline_cer": baseline_cer,
        "final_prediction": best_prediction,
        "final_cer": best_cer,
        "history": history,
    }


def tensor_to_mask_tensor(wm_mask_np, target_shape):
    """watermark.py에서 만든 numpy bool 마스크(H_orig, W_orig)를,
    OcrAttackModel.image_to_tensor()가 만드는 (1,1,32,100) 텐서 크기에 맞춰
    리사이즈한 뒤 torch 텐서(0.0/1.0)로 변환한다.

    주의: 이미지 텐서로 변환할 때(image_to_tensor) 원본 이미지는 비율을 유지하며
    높이 32로 리사이즈되고 오른쪽에 패딩이 붙는다. 마스크도 동일한 변환을
    거쳐야 픽셀 위치가 서로 맞는다. 이 함수는 "이미 image_to_tensor와 동일한
    방식으로 리사이즈/패딩된 마스크 배열"을 받는 것을 전제로 한다.
    """
    import torch as _torch

    arr = wm_mask_np.astype(np.float32)
    tensor = _torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    if tensor.shape[-2:] != target_shape[-2:]:
        tensor = _torch.nn.functional.interpolate(
            tensor, size=target_shape[-2:], mode="nearest"
        )
    return tensor
