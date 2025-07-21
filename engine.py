# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Dict, Iterable, Optional

import torch
import torch.nn
import torch.optim

import util.dist as dist
from datasets.clevrref import ClevrRefEvaluator
from datasets.coco_eval import CocoEvaluator
from datasets.flickr_eval import FlickrEvaluator
from datasets.phrasecut_eval import PhrasecutEvaluator
from datasets.refexp import RefExpEvaluator
from util.metrics import MetricLogger, SmoothedValue
from util.misc import targets_to
from util.optim import adjust_learning_rate, update_ema

# 학습 루프 함수: 입력받은 model, criterion, optimizer, data_loader 등을 통해 1 epoch의 학습 완결
def train_one_epoch(
    model: torch.nn.Module,
    criterion: Optional[torch.nn.Module],
    contrastive_criterion: Optional[torch.nn.Module],
    qa_criterion: Optional[torch.nn.Module],
    weight_dict: Dict[str, float],
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args,
    max_norm: float = 0,
    model_ema: Optional[torch.nn.Module] = None,
):
    # 모델과 사용되는 loss 함수들을 train 모드로 전환하고, 학습 도중 각종 지표(lr, loss, 등)를 기록할 logger 초기화
    model.train()
    if criterion is not None:
        criterion.train()
    if contrastive_criterion is not None:
        contrastive_criterion.train()
    if qa_criterion is not None:
        qa_criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("lr_backbone", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("lr_text_encoder", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    num_training_steps = int(len(data_loader) * args.epochs)
    # data_loader를 통해 배치 단위로 데이터를 가져옴
    for i, batch_dict in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        curr_step = epoch * len(data_loader) + i
        
        # 이미지, 정답 박스 정보, 캡션 등 필수 데이터를 GPU로 이동시키고 정리
        samples = batch_dict["samples"].to(device)
        positive_map = batch_dict["positive_map"].to(device) if "positive_map" in batch_dict else None
        targets = batch_dict["targets"]
        answers = {k: v.to(device) for k, v in batch_dict["answers"].items()} if "answers" in batch_dict else None
        captions = [t["caption"] for t in targets]

        targets = targets_to(targets, device)

        memory_cache = None
        # 모델 forward (인코딩 + 디코딩)
        # encode_and_save=True -> 인코딩 수행 & memory_cache 저장
        # encode_and_save=False -> memory_cache 기반으로 디코딩 수행
        if args.masks: # Segmentation 학습의 경우, 인코딩, 디코딩, 마스크 예측까지 전부 포함
            outputs = model(samples, captions) # DETRsegm.forward(samples, captions) 호출
        else:
            memory_cache = model(samples, captions, encode_and_save=True) # MDETR.forward(samples, captions, ...) 호출 -> 인코딩
            outputs = model(samples, captions, encode_and_save=False, memory_cache=memory_cache) # MDETR.forward(samples, captions, ...) 호출 -> 디코딩

        # 로스 항목별로 계산 후 가중치를 적용해서 전체 로스 산출
        loss_dict = {}
        if criterion is not None:
            loss_dict.update(criterion(outputs, targets, positive_map))

        if contrastive_criterion is not None:
            assert memory_cache is not None
            contrastive_loss = contrastive_criterion(memory_cache["text_pooled_op"], memory_cache["img_pooled_op"])
            loss_dict["contrastive_loss"] = contrastive_loss

        if qa_criterion is not None:
            answer_losses = qa_criterion(outputs, answers)
            loss_dict.update(answer_losses)

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = dist.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # 역전파 및 옵티마이저 업데이트
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        # learning rate 스케줄링 & EMA 업데이트
        adjust_learning_rate(
            optimizer,
            epoch,
            curr_step,
            num_training_steps=num_training_steps,
            args=args,
        )
        if model_ema is not None:
            update_ema(model, model_ema, args.ema_decay)

        # 로깅 & 통계 출력
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(lr_backbone=optimizer.param_groups[1]["lr"])
        metric_logger.update(lr_text_encoder=optimizer.param_groups[2]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# 평가 함수 (학습된 모델을 검증/테스트 데이터셋에 대해 평가)
@torch.no_grad() # gradient 계산 X
def evaluate(
    model: torch.nn.Module,
    criterion: Optional[torch.nn.Module],
    contrastive_criterion: Optional[torch.nn.Module],
    qa_criterion: Optional[torch.nn.Module],
    postprocessors: Dict[str, torch.nn.Module],
    weight_dict: Dict[str, float],
    data_loader,
    evaluator_list,
    device: torch.device,
    args,
):
    # 모든 모듈을 eval 모드로 전환
    model.eval()
    if criterion is not None:
        criterion.eval()
    if contrastive_criterion is not None:
        contrastive_criterion.eval()
    if qa_criterion is not None:
        qa_criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    # 데이터셋 순회 (배치 단위 평가 루프)
    for batch_dict in metric_logger.log_every(data_loader, 10, header):
        # 배치 데이터 준비
        samples = batch_dict["samples"].to(device)
        positive_map = batch_dict["positive_map"].to(device) if "positive_map" in batch_dict else None
        targets = batch_dict["targets"]
        answers = {k: v.to(device) for k, v in batch_dict["answers"].items()} if "answers" in batch_dict else None
        captions = [t["caption"] for t in targets]

        targets = targets_to(targets, device)

        memory_cache = None
        # 모델 forward (인코딩 + 디코딩)
        # encode_and_save=True -> 인코딩 수행 & memory_cache 저장
        # encode_and_save=False -> memory_cache 기반으로 디코딩 수행
        if args.masks: # Segmentation 학습의 경우, 인코딩, 디코딩, 마스크 예측까지 전부 포함
            outputs = model(samples, captions) # DETRsegm.forward(samples, captions) 호출
        else:
            memory_cache = model(samples, captions, encode_and_save=True) # MDETR.forward(samples, captions, ...) 호출 -> 인코딩
            outputs = model(samples, captions, encode_and_save=False, memory_cache=memory_cache) # MDETR.forward(samples, captions, ...) 호출 -> 디코딩

        # loss 계산 (optional)
        loss_dict = {}
        if criterion is not None:
            loss_dict.update(criterion(outputs, targets, positive_map))

        if contrastive_criterion is not None:
            assert memory_cache is not None
            contrastive_loss = contrastive_criterion(memory_cache["text_pooled_op"], memory_cache["img_pooled_op"])
            loss_dict["contrastive_loss"] = contrastive_loss

        if qa_criterion is not None:
            answer_losses = qa_criterion(outputs, answers)
            loss_dict.update(answer_losses)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = dist.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
        metric_logger.update(
            loss=sum(loss_dict_reduced_scaled.values()),
            **loss_dict_reduced_scaled,
            **loss_dict_reduced_unscaled,
        )

        # 예측 결과 후처리
        if not args.no_detection:
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessors["bbox"](outputs, orig_target_sizes)
            if "segm" in postprocessors.keys():
                target_sizes = torch.stack([t["size"] for t in targets], dim=0)
                results = postprocessors["segm"](results, outputs, orig_target_sizes, target_sizes)

            flickr_res = [] if "flickr_bbox" in postprocessors.keys() else None
            if "flickr_bbox" in postprocessors.keys():
                image_ids = [t["original_img_id"] for t in targets]
                sentence_ids = [t["sentence_id"] for t in targets]
                items_per_batch_element = [t["nb_eval"] for t in targets]
                positive_map_eval = batch_dict["positive_map_eval"].to(device)
                flickr_results = postprocessors["flickr_bbox"](
                    outputs, orig_target_sizes, positive_map_eval, items_per_batch_element
                )
                assert len(flickr_results) == len(image_ids) == len(sentence_ids)
                for im_id, sent_id, output in zip(image_ids, sentence_ids, flickr_results):
                    flickr_res.append({"image_id": im_id, "sentence_id": sent_id, "boxes": output})

            phrasecut_res = None
            if "phrasecut" in postprocessors.keys():
                phrasecut_res = postprocessors["phrasecut"](results)
                assert len(targets) == len(phrasecut_res)
                for i in range(len(targets)):
                    phrasecut_res[i]["original_id"] = targets[i]["original_id"]
                    phrasecut_res[i]["task_id"] = targets[i]["task_id"]

            res = {target["image_id"].item(): output for target, output in zip(targets, results)}

            # evaluator에 결과 업데이트
            for evaluator in evaluator_list:
                if isinstance(evaluator, FlickrEvaluator):
                    evaluator.update(flickr_res)
                elif isinstance(evaluator, PhrasecutEvaluator):
                    evaluator.update(phrasecut_res)
                else:
                    evaluator.update(res)

    # 결과 통합 및 요약
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    for evaluator in evaluator_list:
        evaluator.synchronize_between_processes()

    refexp_res = None
    flickr_res = None
    phrasecut_res = None
    for evaluator in evaluator_list:
        if isinstance(evaluator, CocoEvaluator):
            evaluator.accumulate()
            evaluator.summarize()

        elif isinstance(evaluator, (RefExpEvaluator, ClevrRefEvaluator)):
            refexp_res = evaluator.summarize()
        elif isinstance(evaluator, FlickrEvaluator):
            flickr_res = evaluator.summarize()
        elif isinstance(evaluator, PhrasecutEvaluator):
            phrasecut_res = evaluator.summarize()

    # accumulate predictions from all images

    # 평가 결과 정리 및 반환
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    for evaluator in evaluator_list:
        if isinstance(evaluator, CocoEvaluator):
            if "bbox" in postprocessors.keys():
                stats["coco_eval_bbox"] = evaluator.coco_eval["bbox"].stats.tolist()
            if "segm" in postprocessors.keys():
                stats["coco_eval_masks"] = evaluator.coco_eval["segm"].stats.tolist()

    if refexp_res is not None:
        stats.update(refexp_res)

    if flickr_res is not None:
        stats["flickr"] = flickr_res

    if phrasecut_res is not None:
        stats["phrasecut"] = phrasecut_res

    return stats
