# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MDETR model and criterion classes.
"""
from typing import Dict, Optional

import torch
import torch.distributed
import torch.nn.functional as F
from torch import nn

import util.dist as dist
from util import box_ops
from util.metrics import accuracy
from util.misc import NestedTensor, interpolate

from .backbone import build_backbone
from .matcher import build_matcher
from .postprocessors import build_postprocessors
from .segmentation import DETRsegm, dice_loss, sigmoid_focal_loss
from .transformer import build_transformer


class MDETR(nn.Module):
    """ This is the MDETR module that performs modulated object detection """

    # MDETR 모델의 구조를 정의하고 필요한 구성 요소들을 초기화함
    def __init__(
        self,
        backbone,
        transformer,
        num_classes,
        num_queries,
        aux_loss=False,
        contrastive_hdim=64,
        contrastive_loss=False,
        contrastive_align_loss=False,
        qa_dataset: Optional[str] = None,
        split_qa_heads=True,
        predict_final=False,
    ):
        """Initializes the model.

        Args:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         MDETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            contrastive_hdim: dimension used for projecting the embeddings before computing contrastive loss
            contrastive_loss: If true, perform image-text contrastive learning
            contrastive_align_loss: If true, perform box - token contrastive learning
            qa_dataset: If not None, train a QA head for the target dataset (CLEVR or GQA)
            split_qa_heads: If true, use several head for each question type
            predict_final: If true, will predict if a given box is in the actual referred set.
                           Useful for CLEVR-Ref+ only currently.
        """
        super().__init__()
        self.num_queries = num_queries # 고정된 수의 learnable query vector 사용
        self.transformer = transformer # transformer.py에서 정의한 transformer 구조
        hidden_dim = transformer.d_model # transformer 내부 차원
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1) # 각 쿼리에 대해 (클래스 + 배경) 분류 로짓 예측
        # cf. 로짓(logits): softmax나 sigmoid 같은 확률 함수에 들어가기 전의 원시 출력값. 즉, 신경망의 마지막 Linear 층에서 나온 정제되지 않은 스코어.
        self.isfinal_embed = nn.Linear(hidden_dim, 1) if predict_final else None # 여러 박스 중 최종적으로 지칭된 대상을 구별할 때 사용 (CLEVR Ref+용)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3) # 바운딩 박스 4개의 좌표 예측
        self.query_embed = nn.Embedding(num_queries, hidden_dim) # 학습 가능한 Object Query

        if qa_dataset is not None: # QA 태스크용 임베딩 (GQA or CLEVR)
            nb_heads = 6 if qa_dataset == "gqa" else 4
            self.qa_embed = nn.Embedding(nb_heads if split_qa_heads else 1, hidden_dim)

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1) # backbone의 출력 feature map을 transformer가 이해할 수 있도록 채널 수를 맞추는 1x1 Conv
        self.backbone = backbone # backbone.py에서 정의한 backbone 구조
        self.aux_loss = aux_loss # 보조 손실 설정

        # 이미지 전체와 텍스트 전체 간의 대조 학습을 위한 설정
        self.contrastive_loss = contrastive_loss
        if contrastive_loss:
            self.contrastive_projection_image = nn.Linear(hidden_dim, contrastive_hdim, bias=False) # 이미지 전체 표현을 contrastive 학습용으로 projection
            self.contrastive_projection_text = nn.Linear(
                self.transformer.text_encoder.config.hidden_size, contrastive_hdim, bias=False
            ) # 텍스트 전체 표현을 contrastive 학습용으로 projection

        # 이미지 내 각 오브젝트 쿼리와 텍스트의 토큰 간 대조 학습 설정
        self.contrastive_align_loss = contrastive_align_loss
        if contrastive_align_loss:
            self.contrastive_align_projection_image = nn.Linear(hidden_dim, contrastive_hdim) # 각 object query 출력을 같은 차원으로 projection
            self.contrastive_align_projection_text = nn.Linear(hidden_dim, contrastive_hdim) # 텍스트의 각 토큰 표현을 같은 차원으로 projection

        self.qa_dataset = qa_dataset
        self.split_qa_heads = split_qa_heads # QA 태스크용 출력 Head 설정
        if qa_dataset is not None:
            if split_qa_heads:
                self.answer_type_head = nn.Linear(hidden_dim, 5)
                # TODO: make this more general
                if qa_dataset == "gqa": # GQA일 경우
                    self.answer_rel_head = nn.Linear(hidden_dim, 1594)
                    self.answer_obj_head = nn.Linear(hidden_dim, 3)
                    self.answer_global_head = nn.Linear(hidden_dim, 111)
                    self.answer_attr_head = nn.Linear(hidden_dim, 403)
                    self.answer_cat_head = nn.Linear(hidden_dim, 678)
                elif qa_dataset == "clevr": # CLEVR일 경우
                    self.answer_type_head = nn.Linear(hidden_dim, 3)
                    self.answer_binary_head = nn.Linear(hidden_dim, 1)
                    self.answer_attr_head = nn.Linear(hidden_dim, 15)
                    self.answer_reg_head = MLP(hidden_dim, hidden_dim, 20, 3)
                else:
                    assert False, f"Invalid qa dataset {qa_dataset}"
            else:
                # TODO: make this more general
                assert qa_dataset == "gqa", "Clevr QA is not supported with unified head"
                self.answer_head = nn.Linear(hidden_dim, 1853)

    def forward(self, samples: NestedTensor, captions, encode_and_save=True, memory_cache=None):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, height, width). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        # 입력이 NestedTensor가 아니라면, NestedTensor로 변환 NestedTensor: 이미지와 mask를 함께 다루기 위한 MDETR 전용 클래스)
        if not isinstance(samples, NestedTensor):
            samples = NestedTensor.from_tensor_list(samples)

        # encode_and_save=True 모드: 인코딩-only 모드
        # 이미지+텍스트를 인코딩하고 memory_cache를 생성하여 반환
        # 학습 또는 추론 시 디코딩 전에 반드시 먼저 호출해야 함
        if encode_and_save:
            assert memory_cache is None # memory_cache가 None인지 확인 (아니라면 에러 발생)

            features, pos = self.backbone(samples) # backbone으로 이미지 feature, positional encoding 추출
            src, mask = features[-1].decompose() # 가장 마지막 레벨의 feature와 mask만 추출
            query_embed = self.query_embed.weight # 학습 가능한 object query
            if self.qa_dataset is not None: # QA 태스크라면 QA용 query까지 결합
                query_embed = torch.cat([query_embed, self.qa_embed.weight], 0)

            memory_cache = self.transformer(
                self.input_proj(src), # transformer 입력 차원 맞춤
                mask,
                query_embed, # query embedding (num_queries + QA용)
                pos[-1], # positional encoding
                captions, # 텍스트 입력
                encode_and_save=True,
                text_memory=None,
                img_memory=None,
                text_attention_mask=None,
            ) # transformer로 이미지+텍스트 통합 인코딩 + 쿼리 디코딩 수행

            # 이미지, 텍스트의 전체 표현을 같은 차원으로 projection -> 나중에 ContrastiveCriterion에서 이미지-문장 전체 alignment 학습에 사용
            if self.contrastive_loss:
                memory_cache["text_pooled_op"] = self.contrastive_projection_text(memory_cache["text_pooled_op"])
                memory_cache["img_pooled_op"] = self.contrastive_projection_image(memory_cache["img_pooled_op"])

            return memory_cache # transformer 결과를 캐시

        # encode_and_save=False 모드: 디코딩-only 모드 (캐시 기반)
        # encode_and_save=True 모드에서 저장된 기존 memory_cache를 받아서 쿼리 디코딩을 수행
        # 최종 예측 결과를 생성하는 단계
        else:
            assert memory_cache is not None # memory_cache가 존재하는지 확인

            hs = self.transformer(
                mask=memory_cache["mask"],
                query_embed=memory_cache["query_embed"],
                pos_embed=memory_cache["pos_embed"],
                encode_and_save=False,
                text_memory=memory_cache["text_memory_resized"],
                img_memory=memory_cache["img_memory"],
                text_attention_mask=memory_cache["text_attention_mask"],
            ) # memory_cache를 기반으로 transformer에서 디코딩 수행

            out = {}
            if self.qa_dataset is not None: # QA 예측 처리
                if self.split_qa_heads:
                    if self.qa_dataset == "gqa": # GQA의 경우
                        answer_embeds = hs[0, :, -6:] # QA용 쿼리 마지막 6개 추출
                        hs = hs[:, :, :-6] # 나머지는 object detection용
                        out["pred_answer_type"] = self.answer_type_head(answer_embeds[:, 0])
                        out["pred_answer_obj"] = self.answer_obj_head(answer_embeds[:, 1])
                        out["pred_answer_rel"] = self.answer_rel_head(answer_embeds[:, 2])
                        out["pred_answer_attr"] = self.answer_attr_head(answer_embeds[:, 3])
                        out["pred_answer_cat"] = self.answer_cat_head(answer_embeds[:, 4])
                        out["pred_answer_global"] = self.answer_global_head(answer_embeds[:, 5])
                    elif self.qa_dataset == "clevr": # CLEVR의 경우
                        answer_embeds = hs[0, :, -4:] # QA용 쿼리 마지막 4개 추출
                        hs = hs[:, :, :-4] # 나머지는 object detection용
                        out["pred_answer_type"] = self.answer_type_head(answer_embeds[:, 0])
                        out["pred_answer_binary"] = self.answer_binary_head(answer_embeds[:, 1]).squeeze(-1)
                        out["pred_answer_reg"] = self.answer_reg_head(answer_embeds[:, 2])
                        out["pred_answer_attr"] = self.answer_attr_head(answer_embeds[:, 3])
                    else:
                        assert False, f"Invalid qa dataset {self.qa_dataset}"

                else:
                    answer_embeds = hs[0, :, -1]
                    hs = hs[:, :, :-1]
                    out["pred_answer"] = self.answer_head(answer_embeds)

            outputs_class = self.class_embed(hs) # 클래스 로짓
            outputs_coord = self.bbox_embed(hs).sigmoid() # 박스 좌표 (sigmoid로 [정규화)
            out.update(
                {
                    "pred_logits": outputs_class[-1], # 마지막 디코더 층의 클래스 예측
                    "pred_boxes": outputs_coord[-1], # 마지막 디코더 층의 박스 예측
                }
            )

            # CLEVR-Ref+용 isfinal 예측 (선택적)
            outputs_isfinal = None
            if self.isfinal_embed is not None:
                outputs_isfinal = self.isfinal_embed(hs)
                out["pred_isfinal"] = outputs_isfinal[-1]

            # Contrastive Align: 각 object query(hs)와 각 토큰(text_memory)을 같은 차원으로 projection하고 정규화 (선택적)
            proj_queries, proj_tokens = None, None
            if self.contrastive_align_loss:
                proj_queries = F.normalize(self.contrastive_align_projection_image(hs), p=2, dim=-1)
                proj_tokens = F.normalize(
                    self.contrastive_align_projection_text(memory_cache["text_memory"]).transpose(0, 1), p=2, dim=-1
                )
                out.update(
                    {
                        "proj_queries": proj_queries[-1],
                        "proj_tokens": proj_tokens,
                        "tokenized": memory_cache["tokenized"],
                    }
                )

            # Auxiliary Outputs 생성 (보조 손실 계산용)
            if self.aux_loss:
                if self.contrastive_align_loss: # 각 층의 예측값 + 쿼리/토큰 임베딩까지 포함
                    assert proj_tokens is not None and proj_queries is not None
                    out["aux_outputs"] = [
                        {
                            "pred_logits": a,
                            "pred_boxes": b,
                            "proj_queries": c,
                            "proj_tokens": proj_tokens,
                            "tokenized": memory_cache["tokenized"],
                        }
                        for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], proj_queries[:-1])
                    ]
                else: # 각 층의 예측값만
                    out["aux_outputs"] = [
                        {
                            "pred_logits": a,
                            "pred_boxes": b,
                        }
                        for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
                    ]
                if outputs_isfinal is not None: # 각 보조 층에도 isfinal 값을 넣어줌
                    assert len(outputs_isfinal[:-1]) == len(out["aux_outputs"])
                    for i in range(len(outputs_isfinal[:-1])):
                        out["aux_outputs"][i]["pred_isfinal"] = outputs_isfinal[i]
            return out


# 이미지와 텍스트 간의 대조 학습을 위한 손실을 계산하는 클래스
# 글로벌 이미지 <-> 문장 수준의 대응 관계를 학습할 때 사용됨
# 이미지와 텍스트가 서로 매칭되는 경우엔 유사도 높게, 다른 쌍들은 유사도 낮게 학습하도록 유도하는 손실 함수
class ContrastiveCriterion(nn.Module):
    # 초기화
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature # 소프트맥스 스케일링을 조절하는 하이퍼파라미터

    # 주요 연산
    def forward(self, pooled_text, pooled_image):
        # pooled_text: 텍스트 문장 임베딩 / pooled_image: 이미지 전체 임베딩

        # 각 벡터 (텍스트 임베딩, 이미지 임베딩)를 정규화
        normalized_text_emb = F.normalize(pooled_text, p=2, dim=1)
        normalized_img_emb = F.normalize(pooled_image, p=2, dim=1)

        logits = torch.mm(normalized_img_emb, normalized_text_emb.t()) / self.temperature # 두 벡터의 유사도 행렬
        labels = torch.arange(logits.size(0)).to(pooled_image.device) # 정답 레이블 정의 (같은 순서로 정렬되어 있다고 가정)

        loss_i = F.cross_entropy(logits, labels) # 이미지 -> 텍스트 방향 cross-entropy loss
        loss_t = F.cross_entropy(logits.t(), labels) # 텍스트 -> 이미지 방향 cross-entropy loss
        loss = (loss_i + loss_t) / 2.0 # 최종 loss 계산
        return loss


# GQA의 QA 태스크에서의 손실 함수를 정의한 클래스
class QACriterionGQA(nn.Module):
    def __init__(self, split_qa_heads):
        super().__init__()
        self.split_qa_heads = split_qa_heads

    def forward(self, output, answers):
        loss = {}
        if not self.split_qa_heads:
            loss["loss_answer_total"] = F.cross_entropy(output["pred_answer"], answers["answer"], reduction="mean")
            attr_total = (output["pred_answer"].argmax(-1)) == answers["answer"]
            loss["accuracy_answer_total"] = attr_total.float().mean()
            return loss

        device = output["pred_answer_type"].device
        loss["loss_answer_type"] = F.cross_entropy(output["pred_answer_type"], answers["answer_type"])

        type_acc = output["pred_answer_type"].argmax(-1) == answers["answer_type"]
        loss["accuracy_answer_type"] = type_acc.sum() / answers["answer_type"].numel()

        is_obj = answers["answer_type"] == 0
        is_attr = answers["answer_type"] == 1
        is_rel = answers["answer_type"] == 2
        is_global = answers["answer_type"] == 3
        is_cat = answers["answer_type"] == 4

        ## OBJ type
        obj_norm = is_obj.sum() if is_obj.any() else 1.0
        loss["loss_answer_obj"] = (
            F.cross_entropy(output["pred_answer_obj"], answers["answer_obj"], reduction="none")
            .masked_fill(~is_obj, 0)
            .sum()
            / obj_norm
        )
        obj_acc = (output["pred_answer_obj"].argmax(-1)) == answers["answer_obj"]
        loss["accuracy_answer_obj"] = (
            obj_acc[is_obj].sum() / is_obj.sum() if is_obj.any() else torch.as_tensor(1.0, device=device)
        )

        ## ATTR type
        attr_norm = is_attr.sum() if is_attr.any() else 1.0
        loss["loss_answer_attr"] = (
            F.cross_entropy(output["pred_answer_attr"], answers["answer_attr"], reduction="none")
            .masked_fill(~is_attr, 0)
            .sum()
            / attr_norm
        )
        attr_acc = (output["pred_answer_attr"].argmax(-1)) == answers["answer_attr"]
        loss["accuracy_answer_attr"] = (
            attr_acc[is_attr].sum() / is_attr.sum() if is_attr.any() else torch.as_tensor(1.0, device=device)
        )

        ## REL type
        rel_norm = is_rel.sum() if is_rel.any() else 1.0
        loss["loss_answer_rel"] = (
            F.cross_entropy(output["pred_answer_rel"], answers["answer_rel"], reduction="none")
            .masked_fill(~is_rel, 0)
            .sum()
            / rel_norm
        )
        rel_acc = (output["pred_answer_rel"].argmax(-1)) == answers["answer_rel"]
        loss["accuracy_answer_rel"] = (
            rel_acc[is_rel].sum() / is_rel.sum() if is_rel.any() else torch.as_tensor(1.0, device=device)
        )

        ## GLOBAL type
        global_norm = is_global.sum() if is_global.any() else 1.0
        loss["loss_answer_global"] = (
            F.cross_entropy(output["pred_answer_global"], answers["answer_global"], reduction="none")
            .masked_fill(~is_global, 0)
            .sum()
            / global_norm
        )
        global_acc = (output["pred_answer_global"].argmax(-1)) == answers["answer_global"]
        loss["accuracy_answer_global"] = (
            global_acc[is_global].sum() / is_global.sum() if is_global.any() else torch.as_tensor(1.0, device=device)
        )

        ## CAT type
        cat_norm = is_cat.sum() if is_cat.any() else 1.0
        loss["loss_answer_cat"] = (
            F.cross_entropy(output["pred_answer_cat"], answers["answer_cat"], reduction="none")
            .masked_fill(~is_cat, 0)
            .sum()
            / cat_norm
        )
        cat_acc = (output["pred_answer_cat"].argmax(-1)) == answers["answer_cat"]
        loss["accuracy_answer_cat"] = (
            cat_acc[is_cat].sum() / is_cat.sum() if is_cat.any() else torch.as_tensor(1.0, device=device)
        )

        loss["accuracy_answer_total"] = (
            type_acc
            * (is_obj * obj_acc + is_rel * rel_acc + is_attr * attr_acc + is_global * global_acc + is_cat * cat_acc)
        ).sum() / type_acc.numel()

        return loss


# CLEVR의 QA 태스크에서의 손실 함수를 정의한 클래스
class QACriterionClevr(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, answers):
        loss = {}
        loss["loss_answer_type"] = F.cross_entropy(output["pred_answer_type"], answers["answer_type"])

        type_acc = output["pred_answer_type"].argmax(-1) == answers["answer_type"]
        loss["accuracy_answer_type"] = type_acc.sum() / answers["answer_type"].numel()

        is_binary = answers["answer_type"] == 0
        is_attr = answers["answer_type"] == 1
        is_reg = answers["answer_type"] == 2

        binary_norm = is_binary.sum() if is_binary.any() else 1.0
        loss["loss_answer_binary"] = (
            F.binary_cross_entropy_with_logits(output["pred_answer_binary"], answers["answer_binary"], reduction="none")
            .masked_fill(~is_binary, 0)
            .sum()
            / binary_norm
        )
        bin_acc = (output["pred_answer_binary"].sigmoid() > 0.5) == answers["answer_binary"]
        loss["accuracy_answer_binary"] = (
            bin_acc[is_binary].sum() / is_binary.sum() if is_binary.any() else torch.as_tensor(1.0)
        )

        reg_norm = is_reg.sum() if is_reg.any() else 1.0
        loss["loss_answer_reg"] = (
            F.cross_entropy(output["pred_answer_reg"], answers["answer_reg"], reduction="none")
            .masked_fill(~is_reg, 0)
            .sum()
            / reg_norm
        )
        reg_acc = (output["pred_answer_reg"].argmax(-1)) == answers["answer_reg"]
        loss["accuracy_answer_reg"] = reg_acc[is_reg].sum() / is_reg.sum() if is_reg.any() else torch.as_tensor(1.0)

        attr_norm = is_attr.sum() if is_attr.any() else 1.0
        loss["loss_answer_attr"] = (
            F.cross_entropy(output["pred_answer_attr"], answers["answer_attr"], reduction="none")
            .masked_fill(~is_attr, 0)
            .sum()
            / attr_norm
        )
        attr_acc = (output["pred_answer_attr"].argmax(-1)) == answers["answer_attr"]
        loss["accuracy_answer_attr"] = (
            attr_acc[is_attr].sum() / is_attr.sum() if is_attr.any() else torch.as_tensor(1.0)
        )

        loss["accuracy_answer_total"] = (
            type_acc * (is_binary * bin_acc + is_reg * reg_acc + is_attr * attr_acc)
        ).sum() / type_acc.numel()

        return loss


# MDETR의 통합 손실 계산 클래스 (DETR의 Set Prediction 손실 클래스를 확장함)
# cf. MDETR의 세가지 Loss: Set Prediction Loss, Soft Token Prediction Loss, Contrastive Alignment Loss
class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, eos_coef, losses, temperature):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes # 학습 클래스 개수 (no-object 제외)
        self.matcher = matcher # 예측과 Ground Truth를 매칭하는 Hungarian matcher
        self.eos_coef = eos_coef # no-object 클래스에 주는 손실 비중
        self.losses = losses # 손실 항목 리스트
        self.temperature = temperature # softmax 온도 값
        empty_weight = torch.ones(self.num_classes + 1) # + 1: no-object 클래스
        empty_weight[-1] = self.eos_coef # no-object 클래스에만 별도의 손실 가중치 eos_coef 적용
        self.register_buffer("empty_weight", empty_weight)

    # 최종적으로 지목된 객체가 어떤 것인지 예측하는 binary classification loss 계산 함수 (CLEVR-Ref+)
    # 즉, 각 박스가 최종 대상인지 여부를 학습
    def loss_isfinal(self, outputs, targets, positive_map, indices, num_boxes):
        """This loss is used in some referring expression dataset (specifically Clevr-REF+)
        It trains the model to predict which boxes are being referred to (ie are "final")
        Eg if the caption is "the cube next to the cylinder", MDETR will detect both the cube and the cylinder.
        However, the cylinder is an intermediate reasoning step, only the cube is being referred here.
        """
        idx = self._get_src_permutation_idx(indices) # 매칭된 예측 인덱스
        src_isfinal = outputs["pred_isfinal"][idx].squeeze(-1) # 예측된 최종 객체 여부 로짓값
        target_isfinal = torch.cat([t["isfinal"][i] for t, (_, i) in zip(targets, indices)], dim=0) # 최종 객체 여부 정답(타겟)

        loss_isfinal = F.binary_cross_entropy_with_logits(src_isfinal, target_isfinal, reduction="none") # Binary Cross Entropy 손실 계산

        losses = {}
        losses["loss_isfinal"] = loss_isfinal.sum() / num_boxes # 평균 손실 계산
        acc = (src_isfinal.sigmoid() > 0.5) == (target_isfinal > 0.5) # 정확도 계산
        if acc.numel() == 0:
            acc = acc.sum()
        else:
            acc = acc.float().mean()
        losses["accuracy_isfinal"] = acc

        return losses

    # 각 object query가 어떤 텍스트 토큰(span)에 해당하는지를 예측하는 soft classification loss 계산 함수
    # => Soft Token Prediction Loss 계산 함수
    def loss_labels(self, outputs, targets, positive_map, indices, num_boxes):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """

        # 각 object query가 어떤 토큰에 대응되는지를 예측한 값을 log-probability 형태로 변환
        logits = outputs["pred_logits"].log_softmax(-1)  # BS x (num_queries) x (num_tokens)

        src_idx = self._get_src_permutation_idx(indices) # Hungarian Matching 결과에서 예측된 쿼리의 인덱스들을 추출
        tgt_idx = [] # 타켓(정답) 인덱스 벡터
        offset = 0
        for i, (_, tgt) in enumerate(indices):
            tgt_idx.append(tgt + offset)
            offset += len(targets[i]["boxes"])
        tgt_idx = torch.cat(tgt_idx)

        tgt_pos = positive_map[tgt_idx] # 각 정답 객체가 어떤 토큰에 해당되는지 정답 분포
        target_sim = torch.zeros_like(logits)
        target_sim[:, :, -1] = 1
        target_sim[src_idx] = tgt_pos

        loss_ce = -(logits * target_sim).sum(-1) # Soft Cross-Entropy Loss 계산

        eos_coef = torch.full(loss_ce.shape, self.eos_coef, device=target_sim.device)
        eos_coef[src_idx] = 1 # no-object 클래스에 대한 가중치 적용

        loss_ce = loss_ce * eos_coef
        loss_ce = loss_ce.sum() / num_boxes

        losses = {"loss_ce": loss_ce} # 전체 Loss

        return losses

    # 이미지 object query와 텍스트 토큰의 임베딩이 의미상 일치하도록 학습하기 위한 contrastive loss 계산 함수
    # => Contrastive Alignment Loss 계산 함수
    def loss_contrastive_align(self, outputs, targets, positive_map, indices, num_boxes):
        bs = outputs["proj_queries"].shape[0]
        tokenized = outputs["tokenized"]

        # 정규화된 텍스트 임베딩
        normalized_text_emb = outputs["proj_tokens"]  # BS x (num_tokens) x hdim
        # 정규화된 이미지 임베딩
        normalized_img_emb = outputs["proj_queries"]  # BS x (num_queries) x hdim

        # 두 벡터 사이의 유사도 행렬
        logits = (
            torch.matmul(normalized_img_emb, normalized_text_emb.transpose(-1, -2)) / self.temperature
        )  # BS x (num_queries) x (num_tokens)

        # construct a map such that positive_map[k, i,j] = True iff query i is associated to token j in batch item k
        # For efficency, the construction happens on CPU, then the whole matrix is transferred to GPU in one go.
        positive_map = torch.zeros(logits.shape, dtype=torch.bool) # 어떤 query가 어떤 token과 대응되는지를 나타내는 positive_map
        for i, ((idx_src, idx_tgt), tgt) in enumerate(zip(indices, targets)):
            if "tokens_positive" in tgt:
                cur_tokens = [tgt["tokens_positive"][j] for j in idx_tgt]
            else:
                cur_tokens = [tgt["tokens"][j] for j in idx_tgt]

            for j, tok_list in enumerate(cur_tokens):
                for (beg, end) in tok_list: # 각 object query(idx_src[j])가 대응되는 텍스트 범위(beg:end)를 찾아
                    beg_pos = tokenized.char_to_token(i, beg)
                    end_pos = tokenized.char_to_token(i, end - 1)
                    if beg_pos is None:
                        try:
                            beg_pos = tokenized.char_to_token(beg + 1)
                            if beg_pos is None:
                                beg_pos = tokenized.char_to_token(beg + 2)
                        except:
                            beg_pos = None
                    if end_pos is None:
                        try:
                            end_pos = tokenized.char_to_token(end - 2)
                            if end_pos is None:
                                end_pos = tokenized.char_to_token(end - 3)
                        except:
                            end_pos = None
                    if beg_pos is None or end_pos is None:
                        continue

                    assert beg_pos is not None and end_pos is not None
                    positive_map[i, idx_src[j], beg_pos : end_pos + 1].fill_(True) # 해당 query와 token span 사이를 True로 표시 (positive pair 표시)

        positive_map = positive_map.to(logits.device)
        positive_logits = -logits.masked_fill(~positive_map, 0)
        negative_logits = logits  # .masked_fill(positive_map, -1000000)

        # 박스 -> 토큰 방향 loss 계산
        boxes_with_pos = positive_map.any(2)
        pos_term = positive_logits.sum(2)
        neg_term = negative_logits.logsumexp(2)

        nb_pos = positive_map.sum(2) + 1e-6

        box_to_token_loss = ((pos_term / nb_pos + neg_term)).masked_fill(~boxes_with_pos, 0).sum()

        # 토큰 -> 박스 방향 loss 계산
        tokens_with_pos = positive_map.any(1)
        pos_term = positive_logits.sum(1)
        neg_term = negative_logits.logsumexp(1)

        nb_pos = positive_map.sum(1) + 1e-6

        tokens_to_boxes_loss = ((pos_term / nb_pos + neg_term)).masked_fill(~tokens_with_pos, 0).sum()
        
        tot_loss = (box_to_token_loss + tokens_to_boxes_loss) / 2 # 총 loss 계산

        return {"loss_contrastive_align": tot_loss / num_boxes}

    # 예측한 객체 수와 실제 객체 수의 차이를 측정하는 loss 계산 함수 (참고용, 로깅용)
    @torch.no_grad() # 역전파를 하지 않도록 설정
    def loss_cardinality(self, outputs, targets, positive_map, indices, num_boxes):
        """Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs["pred_logits"] # 예측된 로짓
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device) # 각 이미지에 대해 Ground Truth 박스(객체)의 개수 카운트
        ## Count the number of predictions that are NOT "no-object" (which is the last class)
        # normalized_text_emb = outputs["proj_tokens"]  # BS x (num_tokens) x hdim
        # normalized_img_emb = outputs["proj_queries"]  # BS x (num_queries) x hdim

        # logits = torch.matmul(
        #    normalized_img_emb, normalized_text_emb.transpose(-1, -2)
        # )  # BS x (num_queries) x (num_tokens)
        # card_pred = (logits[:, :, 0] > 0.5).sum(1)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1) # 각 이미지에 대해 객체의 개수 카운트 (no-object 클래스 제외)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float()) # L1 loss 계산
        losses = {"cardinality_error": card_err}
        return losses

    # 예측된 박스(바운딩 박스)와 정답 박스 간의 위치 정확도를 평가하기 위해 L1 & GIoU loss 계산 함수
    def loss_boxes(self, outputs, targets, positive_map, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx] # 예측 박스 중 GT와 매칭된 것만 가져옴
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0) # GT 박스 중 매칭된 것들만 추출

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none") # 예측 박스와 정답 박스 간의 L1 loss 계산

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(src_boxes), box_ops.box_cxcywh_to_xyxy(target_boxes))
        ) # GIoU loss 계산
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    # segmentation 마스크에 대해 Focal Loss와 Dice Loss를 계산하는 함수 (선택적)
    def loss_masks(self, outputs, targets, positive_map, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices) # Hungarian matching 결과에서 source(예측 쿼리) 인덱스 추출
        tgt_idx = self._get_tgt_permutation_idx(indices) # Hungarian matching 결과에서 target(정답 박스) 인덱스 추출

        src_masks = outputs["pred_masks"] # 모델이 예측한 마스크 가져옴

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = NestedTensor.from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    # 모델의 예측값에서 매칭된 인덱스를 추출
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    # 정답(타겟)값에서 매칭된 인덱스를 추출
    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    # 특정 loss 이름을 받아서 해당 손실 함수 실행
    def get_loss(self, loss, outputs, targets, positive_map, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
            "isfinal": self.loss_isfinal,
            "contrastive_align": self.loss_contrastive_align,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, positive_map, indices, num_boxes, **kwargs)

    # 모델 출력과 정답 간의 매칭을 통해 모든 손실을 계산하고 반환하는 함수
    def forward(self, outputs, targets, positive_map):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"} # outputs 중에서 aux_outputs을 제외한 최종 출력만 따로 저장

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, positive_map) # Hungarian matching 등을 사용해서 object query와 정답 box를 짝지음

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if dist.is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / dist.get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses: # self.losses에 등록된 손실 항목들을 차례로 계산
            losses.update(self.get_loss(loss, outputs, targets, positive_map, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs: # 중간 출력(aux_outputs)에 대해서도 반복 계산
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets, positive_map)
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, positive_map, indices, num_boxes, **kwargs)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


# 다층 퍼셉트론(Multi-Layer Perceptron, MLP) 또는 Feed-Forward Network (FFN) 클래스
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# MDETR 모델과 해당 구성 옵션에 따라 손실 함수 및 weight dict을 초기화하는 빌더 함수
def build(args):
    num_classes = 255
    device = torch.device(args.device)

    assert not args.masks or args.mask_model != "none"

    qa_dataset = None
    if args.do_qa:
        assert not (
            ("clevr" in args.combine_datasets or "clevr_question" in args.combine_datasets)
            and "gqa" in args.combine_datasets
        ), "training GQA and CLEVR simultaneously is not supported"
        assert (
            "clevr_question" in args.combine_datasets
            or "clevr" in args.combine_datasets
            or "gqa" in args.combine_datasets
        ), "Question answering require either gqa or clevr dataset"
        qa_dataset = "gqa" if "gqa" in args.combine_datasets else "clevr"

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = MDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        contrastive_hdim=args.contrastive_loss_hdim,
        contrastive_loss=args.contrastive_loss,
        contrastive_align_loss=args.contrastive_align_loss,
        qa_dataset=qa_dataset,
        split_qa_heads=args.split_qa_heads,
        predict_final=args.predict_final,
    )
    if args.mask_model != "none":
        model = DETRsegm(
            model,
            mask_head=args.mask_model,
            freeze_detr=(args.frozen_weights is not None),
        )
    matcher = build_matcher(args)
    weight_dict = {"loss_ce": args.ce_loss_coef, "loss_bbox": args.bbox_loss_coef}
    if args.contrastive_loss:
        weight_dict["contrastive_loss"] = args.contrastive_loss_coef
    if args.contrastive_align_loss:
        weight_dict["loss_contrastive_align"] = args.contrastive_align_loss_coef
    if args.predict_final:
        weight_dict["loss_isfinal"] = 1

    weight_dict["loss_giou"] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef

    if args.do_qa:
        if args.split_qa_heads:
            weight_dict["loss_answer_type"] = 1 * args.qa_loss_coef
            if qa_dataset == "gqa":
                weight_dict["loss_answer_cat"] = 1 * args.qa_loss_coef
                weight_dict["loss_answer_attr"] = 1 * args.qa_loss_coef
                weight_dict["loss_answer_rel"] = 1 * args.qa_loss_coef
                weight_dict["loss_answer_obj"] = 1 * args.qa_loss_coef
                weight_dict["loss_answer_global"] = 1 * args.qa_loss_coef
            else:
                weight_dict["loss_answer_binary"] = 1
                weight_dict["loss_answer_attr"] = 1
                weight_dict["loss_answer_reg"] = 1

        else:
            weight_dict["loss_answer_total"] = 1 * args.qa_loss_coef

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ["labels", "boxes", "cardinality"]
    if args.masks:
        losses += ["masks"]
    if args.predict_final:
        losses += ["isfinal"]
    if args.contrastive_align_loss:
        losses += ["contrastive_align"]

    criterion = None
    if not args.no_detection:
        criterion = SetCriterion(
            num_classes,
            matcher=matcher,
            eos_coef=args.eos_coef,
            losses=losses,
            temperature=args.temperature_NCE,
        )
        criterion.to(device)

    if args.contrastive_loss:
        contrastive_criterion = ContrastiveCriterion(temperature=args.temperature_NCE)
        contrastive_criterion.to(device)
    else:
        contrastive_criterion = None

    if args.do_qa:
        if qa_dataset == "gqa":
            qa_criterion = QACriterionGQA(split_qa_heads=args.split_qa_heads)
        elif qa_dataset == "clevr":
            qa_criterion = QACriterionClevr()
        else:
            assert False, f"Invalid qa dataset {qa_dataset}"
        qa_criterion.to(device)
    else:
        qa_criterion = None
    return model, criterion, contrastive_criterion, qa_criterion, weight_dict
