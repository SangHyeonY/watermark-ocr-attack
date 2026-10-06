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

    Args:
        ocr_wrapper, original_tensor, wm_mask_tensor: attack_watermark_region과 동일.
        original_text: 이 이미지의 실제(정답) 텍스트. 공격 목표가 아니라
            "이걸로 읽히면 공격 실패"라는 기준점으로 쓰인다.
        eps, eps_iter, max_iter, verbose: attack_watermark_region과 동일.
            eps 기본값(0.45)은 5개 한글 단어로 0.3~0.6 구간을 비교해서 정한 값으로,
            "공격 성공률 100%를 유지하는 가장 작은 값"이다. eps가 작을수록 워터마크
            변화가 연해져 사람 눈에 덜 거슬리지만, 너무 작으면(0.3~0.4) 일부 단어에서
            공격이 실패했다 (WORK_LOG.md 6단계 참고). 단어 길이/폰트/문장 단위로
            확장하면 이 값은 재조정이 필요할 수 있는 잠정값이다.

    Returns:
        dict with keys:
            "adv_tensor": 최종 적대적 이미지 텐서
            "success": bool, 최종 예측이 original_text와 달라졌는지 여부
            "success_iter": 성공한 iteration 번호 (실패 시 None)
            "final_prediction": 마지막 iteration에서의 OCR 예측 문자열
            "history": 각 iteration의 (prediction, ctc_loss) 리스트
    """
    clip_min, clip_max = -1.0, 1.0

    adv_tensor = original_tensor.clone().detach()
    momentum = torch.zeros_like(adv_tensor)

    success = False
    success_iter = None
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
        history.append((prediction, loss.item()))

        if verbose and (it % 20 == 0 or it == max_iter - 1):
            print(f"  [iter {it:3d}] CTC loss(정답 기준)={loss.item():.4f}, 현재 예측='{prediction}'")

        if prediction != original_text and not success:
            success = True
            success_iter = it
            if verbose:
                print(f"  -> 공격 성공! iteration {it}에서 더 이상 '{original_text}'로 읽히지 않음 (현재: '{prediction}')")
            break

    return {
        "adv_tensor": adv_tensor.detach(),
        "success": success,
        "success_iter": success_iter,
        "final_prediction": history[-1][0] if history else None,
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
