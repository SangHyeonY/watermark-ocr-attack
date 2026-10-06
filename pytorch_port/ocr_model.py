# -*- coding: utf-8 -*-
"""
EasyOCR의 인식(recognition) 모델을 감싸서, 공격(적대적 섭동) 코드에서 쓰기 쉽게
"이미지 텐서 -> CTC loss, gradient" 흐름을 제공하는 래퍼.

왜 필요한가:
- EasyOCR은 기본적으로 reader.readtext(이미지)처럼 "그림 파일 -> 최종 글자"를
  한번에 처리해주는 高수준 API만 제공한다. 이 API는 내부적으로 PIL 이미지 변환,
  디코딩(최댓값 뽑기) 등을 포함하는데, 이 과정에는 미분(gradient)이 흐르지 않는
  연산이 섞여 있어서 "입력 이미지를 아주 조금 바꾸면 결과가 어떻게 바뀔지"를
  역전파로 계산할 수 없다.
- FAWA 공격은 "OCR이 틀리게 읽도록 유도하는 손실(loss)을 계산하고, 그 손실을
  입력 이미지 픽셀까지 역전파해서 '어느 방향으로 픽셀을 바꾸면 더 틀리게
  만들 수 있는지' 알아내는" 방식이다. 그러려면 저수준에서
  "전처리된 이미지 텐서 -> 모델 forward -> CTC loss"까지 전부 PyTorch
  연산(미분 가능)으로 연결되어 있어야 한다.
- 이 파일은 그 저수준 흐름만 노출하는 래퍼 클래스를 제공한다.

중요 발견 (WORK_LOG.md 5단계 참고):
- easyocr.Reader(gpu=False)의 기본값은 quantize=True 인데, 이는 모델을
  int8로 압축(양자화)해서 속도를 높이는 처리이다. 양자화된 모델은
  PyTorch autograd(자동 미분)를 지원하지 않아서 loss.backward()를 호출해도
  입력 이미지의 .grad가 None으로 나온다. 따라서 공격용으로 Reader를 만들 때는
  반드시 quantize=False로 설정해야 한다.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# EasyOCR 인식 모델이 기대하는 입력 사양 (recognition.py AlignCollate 참고)
IMG_HEIGHT = 32
IMG_WIDTH = 100
# batch_max_length는 EasyOCR 내부에서 int(imgW/10)으로 계산되지만,
# 실제 CTCLabelConverter.encode에는 영향 없음 (text_for_pred placeholder용).
BATCH_MAX_LENGTH = int(IMG_WIDTH / 10)


class OcrAttackModel:
    """EasyOCR 인식 모델을 감싸는 공격용 래퍼.

    사용 예:
        wrapper = OcrAttackModel(reader)
        tensor = wrapper.image_to_tensor(pil_image)  # requires_grad=True로 만들어짐
        loss = wrapper.ctc_loss(tensor, target_text="문제")
        loss.backward()
        tensor.grad  # 이제 이 방향으로 픽셀을 바꾸면 손실이 커짐(=더 틀리게 만듦)
    """

    def __init__(self, reader):
        """
        Args:
            reader: easyocr.Reader 인스턴스. 반드시
                easyocr.Reader(..., gpu=False, quantize=False)로 생성해야 한다.
                (quantize=True면 gradient가 전혀 나오지 않음 - 위 모듈 docstring 참고)
        """
        self.model = reader.recognizer
        self.model.eval()  # 추론 모드(dropout 등 비활성화)이지만 gradient 계산은 그대로 가능
        self.converter = reader.converter

    def image_to_tensor(self, pil_image):
        """PIL 이미지를 EasyOCR 인식 모델 입력 형식의 텐서로 변환한다.

        원본 FAWA의 scale() 함수(높이 32로 리사이즈 후 정규화)와 같은 역할.
        EasyOCR의 recognition.py NormalizePAD/AlignCollate 로직을 참고해 재현.

        Returns:
            torch.Tensor, shape (1, 1, 32, 100), requires_grad=True로 설정됨.
            값 범위는 [-1, 1] (EasyOCR 내부 정규화 방식과 동일).
        """
        gray = pil_image.convert("L")
        w, h = gray.size
        ratio = w / float(h)
        resized_w = min(IMG_WIDTH, math.ceil(IMG_HEIGHT * ratio))
        resized = gray.resize((resized_w, IMG_HEIGHT), Image.BICUBIC)

        arr = np.array(resized).astype(np.float32) / 255.0  # [0, 1]
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, resized_w)
        tensor = (tensor - 0.5) / 0.5  # [-1, 1] 정규화 (NormalizePAD와 동일)

        # 오른쪽 패딩 (EasyOCR의 PAD_type='right'와 동일: 마지막 열을 반복해서 채움)
        padded = torch.zeros(1, 1, IMG_HEIGHT, IMG_WIDTH)
        padded[:, :, :, :resized_w] = tensor
        if resized_w < IMG_WIDTH:
            last_col = tensor[:, :, :, -1:].expand(1, 1, IMG_HEIGHT, IMG_WIDTH - resized_w)
            padded[:, :, :, resized_w:] = last_col

        padded.requires_grad_(True)
        return padded

    def mask_to_tensor(self, mask_array, orig_pil_size):
        """numpy 불리언 마스크(H, W)를 image_to_tensor()와 동일한 좌표계로 변환한다.

        watermark.py의 wm_mask/text_mask는 "합성 전 원본 글자 이미지" 크기의
        배열이다. 공격 시에는 image_to_tensor()로 리사이즈+패딩된 이미지 텐서와
        픽셀 위치가 정확히 맞아야 하므로, 동일한 resize 비율과 패딩 방식을
        마스크에도 그대로 적용한다. (패딩 영역은 "워터마크도 글자도 아닌 빈 공간"
        이므로 항상 0으로 채운다 - 마지막 열을 반복하지 않는다는 점이 이미지
        패딩과 다른 부분)

        Args:
            mask_array: np.ndarray(bool 또는 0/1), shape (H, W). watermark.py가
                반환하는 wm_mask 또는 text_mask.
            orig_pil_size: (width, height) - 마스크가 기준으로 하는 원본 PIL 이미지 크기.
                (watermark.py의 composite 이미지 크기와 동일해야 함)

        Returns:
            torch.Tensor, shape (1, 1, 32, 100), 값은 0.0 또는 1.0.
        """
        w, h = orig_pil_size
        ratio = w / float(h)
        resized_w = min(IMG_WIDTH, math.ceil(IMG_HEIGHT * ratio))

        mask_img = Image.fromarray((np.asarray(mask_array).astype(np.uint8)) * 255)
        mask_resized = mask_img.resize((resized_w, IMG_HEIGHT), Image.NEAREST)
        mask_arr = (np.array(mask_resized) > 127).astype(np.float32)

        tensor = torch.from_numpy(mask_arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,resized_w)

        padded = torch.zeros(1, 1, IMG_HEIGHT, IMG_WIDTH)
        padded[:, :, :, :resized_w] = tensor
        # 패딩 영역(resized_w 이후)은 0으로 유지 (워터마크도 글자도 없는 빈 공간)
        return padded

    def forward_logits(self, image_tensor):
        """이미지 텐서를 모델에 통과시켜 (batch, seq_len, num_class) 로짓을 얻는다."""
        batch_size = image_tensor.size(0)
        text_for_pred = torch.LongTensor(batch_size, BATCH_MAX_LENGTH + 1).fill_(0)
        return self.model(image_tensor, text_for_pred)

    def ctc_loss(self, image_tensor, target_text):
        """주어진 이미지 텐서가 target_text로 인식되도록 만드는 CTC 손실을 계산한다.

        공격 시 사용법:
        - "OCR이 target_text(공격 목표 텍스트, 예: 엉뚱한 단어)로 읽도록" 손실을
          낮추는 방향 = 공격이 성공하는 방향.
        - 즉 이 손실을 역전파해서 얻은 gradient의 "음의 방향(-gradient)"으로
          이미지를 조금씩 수정하면, target_text로 인식될 확률이 높아진다
          (= 원래 글자가 아닌 다른 글자로 잘못 읽게 됨 = 공격 성공).

        Args:
            image_tensor: self.image_to_tensor()로 만든 텐서 (requires_grad=True 권장)
            target_text: 공격 목표 문자열 (예: 원본이 "문제"면 "무지" 같은 엉뚱한 문자열)

        Returns:
            scalar tensor (CTC loss 값). .backward() 호출 가능.
        """
        preds = self.forward_logits(image_tensor)
        log_probs = F.log_softmax(preds, dim=2).permute(1, 0, 2)  # (T, N, C)

        target_index, target_length = self.converter.encode([target_text])
        input_length = torch.IntTensor([preds.size(1)] * image_tensor.size(0))

        loss = F.ctc_loss(log_probs, target_index, input_length, target_length)
        return loss

    @torch.no_grad()
    def predict_text(self, image_tensor):
        """현재 이미지 텐서가 어떤 텍스트로 인식되는지 그리디 디코딩으로 확인한다.

        공격 진행 중 "지금 몇 iteration째인데 아직 원래 글자로 읽히는지,
        목표 글자로 바뀌었는지"를 확인하는 용도.
        """
        preds = self.forward_logits(image_tensor)
        preds_size = torch.IntTensor([preds.size(1)] * image_tensor.size(0))
        _, preds_index = preds.max(2)
        preds_index = preds_index.view(-1)
        texts = self.converter.decode_greedy(preds_index.data.cpu().numpy(), preds_size.data)
        return texts[0] if texts else ""


def tensor_to_pil_image(image_tensor):
    """OCR 모델 입력용 텐서([-1,1] 범위, (1,1,32,100))를 사람이 눈으로 볼 수 있는
    흑백 PIL 이미지로 되돌린다.

    image_to_tensor()에서 한 정규화(값을 0.5 빼고 0.5로 나눔)의 역변환
    (0.5를 곱하고 0.5를 더함)을 수행한다. 공격 전/후 이미지를 나란히 저장해서
    눈으로 비교하기 위한 용도.
    """
    arr = image_tensor.detach().cpu().numpy()[0, 0]  # (32, 100)
    arr = (arr * 0.5 + 0.5) * 255.0  # [-1,1] -> [0,255]
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")
