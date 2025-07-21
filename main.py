# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import os
import random
import time
from collections import namedtuple
from copy import deepcopy
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.utils
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler

import util.dist as dist
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from datasets.clevrref import ClevrRefEvaluator
from datasets.coco_eval import CocoEvaluator
from datasets.flickr_eval import FlickrEvaluator
from datasets.phrasecut_eval import PhrasecutEvaluator
from datasets.refexp import RefExpEvaluator
from engine import evaluate, train_one_epoch
from models import build_model
from models.postprocessors import build_postprocessors


# 커맨드라인 인자들을 정의하는 함수
def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--run_name", default="", type=str)

    # Dataset specific
    parser.add_argument("--dataset_config", default=None, required=True)
    parser.add_argument("--do_qa", action="store_true", help="Whether to do question answering")
    parser.add_argument(
        "--predict_final",
        action="store_true",
        help="If true, will predict if a given box is in the actual referred set. Useful for CLEVR-Ref+ only currently.",
    )
    parser.add_argument("--no_detection", action="store_true", help="Whether to train the detector")
    parser.add_argument(
        "--split_qa_heads", action="store_true", help="Whether to use a separate head per question type in vqa"
    )
    parser.add_argument(
        "--combine_datasets", nargs="+", help="List of datasets to combine for training", default=["flickr"]
    )
    parser.add_argument(
        "--combine_datasets_val", nargs="+", help="List of datasets to combine for eval", default=["flickr"]
    )

    parser.add_argument("--coco_path", type=str, default="")
    parser.add_argument("--vg_img_path", type=str, default="")
    parser.add_argument("--vg_ann_path", type=str, default="")
    parser.add_argument("--clevr_img_path", type=str, default="")
    parser.add_argument("--clevr_ann_path", type=str, default="")
    parser.add_argument("--phrasecut_ann_path", type=str, default="")
    parser.add_argument(
        "--phrasecut_orig_ann_path",
        type=str,
        default="",
    )
    parser.add_argument("--modulated_lvis_ann_path", type=str, default="")

    # Training hyper-parameters
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_backbone", default=1e-5, type=float)
    parser.add_argument("--text_encoder_lr", default=5e-5, type=float)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--lr_drop", default=35, type=int)
    parser.add_argument(
        "--epoch_chunks",
        default=-1,
        type=int,
        help="If greater than 0, will split the training set into chunks and validate/checkpoint after each chunk",
    )
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm")
    parser.add_argument(
        "--eval_skip",
        default=1,
        type=int,
        help='do evaluation every "eval_skip" frames',
    )

    parser.add_argument(
        "--schedule",
        default="linear_with_warmup",
        type=str,
        choices=("step", "multistep", "linear_with_warmup", "all_linear_with_warmup"),
    )
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.9998)
    parser.add_argument("--fraction_warmup_steps", default=0.01, type=float, help="Fraction of total number of steps")

    # Model parameters
    parser.add_argument(
        "--frozen_weights",
        type=str,
        default=None,
        help="Path to the pretrained model. If set, only the mask head will be trained",
    )

    parser.add_argument(
        "--freeze_text_encoder", action="store_true", help="Whether to freeze the weights of the text encoder"
    )

    parser.add_argument(
        "--text_encoder_type",
        default="roberta-base",
        choices=("roberta-base", "distilroberta-base", "roberta-large"),
    )

    # Backbone
    parser.add_argument(
        "--backbone",
        default="resnet101",
        type=str,
        help="Name of the convolutional backbone to use such as resnet50 resnet101 timm_tf_efficientnet_b3_ns",
    )
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="If true, we replace stride with dilation in the last convolutional block (DC5)",
    )
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned"),
        help="Type of positional embedding to use on top of the image features",
    )

    # Transformer
    parser.add_argument(
        "--enc_layers",
        default=6,
        type=int,
        help="Number of encoding layers in the transformer",
    )
    parser.add_argument(
        "--dec_layers",
        default=6,
        type=int,
        help="Number of decoding layers in the transformer",
    )
    parser.add_argument(
        "--dim_feedforward",
        default=2048,
        type=int,
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument(
        "--hidden_dim",
        default=256,
        type=int,
        help="Size of the embeddings (dimension of the transformer)",
    )
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument(
        "--nheads",
        default=8,
        type=int,
        help="Number of attention heads inside the transformer's attentions",
    )
    parser.add_argument("--num_queries", default=100, type=int, help="Number of query slots")
    parser.add_argument("--pre_norm", action="store_true")
    parser.add_argument(
        "--no_pass_pos_and_query",
        dest="pass_pos_and_query",
        action="store_false",
        help="Disables passing the positional encodings to each attention layers",
    )

    # Segmentation
    parser.add_argument(
        "--mask_model",
        default="none",
        type=str,
        choices=("none", "smallconv", "v2"),
        help="Segmentation head to be used (if None, segmentation will not be trained)",
    )
    parser.add_argument("--remove_difficult", action="store_true")
    parser.add_argument("--masks", action="store_true")

    # Loss
    parser.add_argument(
        "--no_aux_loss",
        dest="aux_loss",
        action="store_false",
        help="Disables auxiliary decoding losses (loss at each layer)",
    )
    parser.add_argument(
        "--set_loss",
        default="hungarian",
        type=str,
        choices=("sequential", "hungarian", "lexicographical"),
        help="Type of matching to perform in the loss",
    )

    parser.add_argument("--contrastive_loss", action="store_true", help="Whether to add contrastive loss")
    parser.add_argument(
        "--no_contrastive_align_loss",
        dest="contrastive_align_loss",
        action="store_false",
        help="Whether to add contrastive alignment loss",
    )

    parser.add_argument(
        "--contrastive_loss_hdim",
        type=int,
        default=64,
        help="Projection head output size before computing normalized temperature-scaled cross entropy loss",
    )

    parser.add_argument(
        "--temperature_NCE", type=float, default=0.07, help="Temperature in the  temperature-scaled cross entropy loss"
    )

    # * Matcher
    parser.add_argument(
        "--set_cost_class",
        default=1,
        type=float,
        help="Class coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_bbox",
        default=5,
        type=float,
        help="L1 box coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_giou",
        default=2,
        type=float,
        help="giou box coefficient in the matching cost",
    )
    # Loss coefficients
    parser.add_argument("--ce_loss_coef", default=1, type=float)
    parser.add_argument("--mask_loss_coef", default=1, type=float)
    parser.add_argument("--dice_loss_coef", default=1, type=float)
    parser.add_argument("--bbox_loss_coef", default=5, type=float)
    parser.add_argument("--giou_loss_coef", default=2, type=float)
    parser.add_argument("--qa_loss_coef", default=1, type=float)
    parser.add_argument(
        "--eos_coef",
        default=0.1,
        type=float,
        help="Relative classification weight of the no-object class",
    )
    parser.add_argument("--contrastive_loss_coef", default=0.1, type=float)
    parser.add_argument("--contrastive_align_loss_coef", default=1, type=float)

    # Run specific

    parser.add_argument("--test", action="store_true", help="Whether to run evaluation on val or test set")
    parser.add_argument("--test_type", type=str, default="test", choices=("testA", "testB", "test"))
    parser.add_argument("--output-dir", default="", help="path where to save, empty for no saving")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--load", default="", help="resume from checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--eval", action="store_true", help="Only run evaluation")
    parser.add_argument("--num_workers", default=5, type=int)

    # Distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    return parser


def main(args):
    # 1단계: 환경 초기화 및 설정 불러오기

    # 분산 학습 설정 (단일 GPU인 경우 내부적으로 무시됨)
    # Init distributed mode
    dist.init_distributed_mode(args)

    # --dataset_config 파일(.json)에 정의된 설정들 로딩
    # Update dataset specific configs
    if args.dataset_config is not None:
        # https://stackoverflow.com/a/16878364
        d = vars(args)
        with open(args.dataset_config, "r") as f:
            cfg = json.load(f)
        d.update(cfg)

    print("git:\n  {}\n".format(utils.get_sha()))

    # segmentation이 포함된 모델인 경우, segmentation 관련 옵션 처리
    # Segmentation related
    if args.mask_model != "none":
        args.masks = True
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"

    # 최종적으로 적용된 모든 설정값(args) 출력
    print(args)

    device = torch.device(args.device) # 학습에 사용할 디바이스 설정
    output_dir = Path(args.output_dir) # 결과 저장할 폴더 경로 지정

    # seed 설정 (분산 학습일 경우 GPU마다 seed를 다르게)
    # fix the seed for reproducibility
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_deterministic(True)

    # 2단계: 모델 구성 및 옵티마이저 설정

    # Build the model
    model, criterion, contrastive_criterion, qa_criterion, weight_dict = build_model(args) # 모델과 손실 함수들 생성
    model.to(device)

    # detection 또는 QA 중 하나는 반드시 학습해야 한다는 조건
    assert (
        criterion is not None or qa_criterion is not None
    ), "Error: should train either detection or question answering (or both)"

    # Get a copy of the model for exponential moving averaged version of the model
    model_ema = deepcopy(model) if args.ema else None # EMA 모델 생성 (선택)
    # 분산 학습 시 DDP (DistributedDataParallel) 래핑
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    # 전체 학습 가능한 파라미터 개수 출력
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    # 옵티마이저 설정
    # 학습 파라미터를 그룹별로 나눠서 각각 다른 learning rate 적용: 일반 파라미터 → args.lr / backbone 관련 → args.lr_backbone / text_encoder 관련 → args.text_encoder_lr
    # Set up optimizers
    param_dicts = [
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if "backbone" not in n and "text_encoder" not in n and p.requires_grad
            ]
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "text_encoder" in n and p.requires_grad],
            "lr": args.text_encoder_lr,
        },
    ]
    # --optimizer 인자에 따라 적절한 optimizer 선택
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer in ["adam", "adamw"]:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Unsupported optimizer {args.optimizer}")

    # 3단계: 학습 & 검증 데이터셋 구성

    # 학습 데이터셋 구성
    # Train dataset
    if len(args.combine_datasets) == 0 and not args.eval:
        raise RuntimeError("Please provide at least one training dataset")

    dataset_train, sampler_train, data_loader_train = None, None, None
    if not args.eval: # --eval이 켜져 있지 않다면 (학습도 한다면) 학습용 데이터셋 준비
        dataset_train = ConcatDataset(
            [build_dataset(name, image_set="train", args=args) for name in args.combine_datasets]
        ) # 여러 데이터셋을 묶어서 사용

        # To handle very big datasets, we chunk it into smaller parts.
        if args.epoch_chunks > 0: # epoch_chunks 설정 시, 데이터셋을 쪼개서 학습
            print(
                "Splitting the training set into {args.epoch_chunks} of size approximately "
                f" {len(dataset_train) // args.epoch_chunks}"
            )
            # 데이터셋 인덱스를 나누고, 각 chunk마다 서브셋을 만들고 RandomSampler or DistributedSampler를 적용
            chunks = torch.chunk(torch.arange(len(dataset_train)), args.epoch_chunks)
            datasets = [torch.utils.data.Subset(dataset_train, chunk.tolist()) for chunk in chunks]
            if args.distributed:
                samplers_train = [DistributedSampler(ds) for ds in datasets]
            else:
                samplers_train = [torch.utils.data.RandomSampler(ds) for ds in datasets]

            # BatchSampler와 함께 DataLoader 생성
            batch_samplers_train = [
                torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
                for sampler_train in samplers_train
            ]
            assert len(batch_samplers_train) == len(datasets)
            data_loaders_train = [
                DataLoader(
                    ds,
                    batch_sampler=batch_sampler_train,
                    collate_fn=partial(utils.collate_fn, False),
                    num_workers=args.num_workers,
                )
                for ds, batch_sampler_train in zip(datasets, batch_samplers_train)
            ]
        else: # 일반적인 경우, 하나의 전체 데이터셋으로 학습
            if args.distributed:
                sampler_train = DistributedSampler(dataset_train)
            else:
                sampler_train = torch.utils.data.RandomSampler(dataset_train)

            batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
            data_loader_train = DataLoader(
                dataset_train,
                batch_sampler=batch_sampler_train,
                collate_fn=partial(utils.collate_fn, False),
                num_workers=args.num_workers,
            )

    # 검증 데이터셋 구성
    # Val dataset
    if len(args.combine_datasets_val) == 0:
        raise RuntimeError("Please provide at leas one validation dataset")

    Val_all = namedtuple(typename="val_data", field_names=["dataset_name", "dataloader", "base_ds", "evaluator_list"])

    val_tuples = []
    for dset_name in args.combine_datasets_val:
        dset = build_dataset(dset_name, image_set="val", args=args)
        sampler = (
            DistributedSampler(dset, shuffle=False) if args.distributed else torch.utils.data.SequentialSampler(dset)
        )
        dataloader = DataLoader(
            dset,
            args.batch_size,
            sampler=sampler,
            drop_last=False,
            collate_fn=partial(utils.collate_fn, False),
            num_workers=args.num_workers,
        )
        base_ds = get_coco_api_from_dataset(dset)
        val_tuples.append(Val_all(dataset_name=dset_name, dataloader=dataloader, base_ds=base_ds, evaluator_list=None))

    # 4단계: 체크포인트 로딩

    # --frozen_weights: frozen segmentation용 weight 불러오기 (segmentation만 학습하고자 할 때 사용)
    # DETR 부분만 불러와서 그대로 사용하고, mask head만 학습합니다
    if args.frozen_weights is not None:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location="cpu", check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")
        if "model_ema" in checkpoint and checkpoint["model_ema"] is not None:
            model_without_ddp.detr.load_state_dict(checkpoint["model_ema"], strict=False)
        else:
            model_without_ddp.detr.load_state_dict(checkpoint["model"], strict=False)

        if args.ema:
            model_ema = deepcopy(model_without_ddp)

    # --load: 전체 모델을 가져오되 새로 학습 시작하기
    # 기존 pretrained 모델을 불러와서 가중치만 가져올 때 사용 (학습은 처음부터 시작) 
    # epoch, optimizer 등은 가져오지 않음
    # Used for loading weights from another model and starting a training from scratch. Especially useful if
    # loading into a model with different functionality.
    if args.load:
        print("loading from", args.load)
        checkpoint = torch.load(args.load, map_location="cpu")
        if "model_ema" in checkpoint:
            model_without_ddp.load_state_dict(checkpoint["model_ema"], strict=False)
        else:
            model_without_ddp.load_state_dict(checkpoint["model"], strict=False)

        if args.ema:
            model_ema = deepcopy(model_without_ddp)

    # --resume: 중단된 학습 이어서 하기
    # 모델 가중치를 그대로 복구 & optimizer 상태, epoch 정보까지 복원해서 이어서 학습
    # Used for resuming training from the checkpoint of a model. Used when training times-out or is pre-empted.
    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location="cpu", check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.eval and "optimizer" in checkpoint and "epoch" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            args.start_epoch = checkpoint["epoch"] + 1
        if args.ema:
            if "model_ema" not in checkpoint:
                print("WARNING: ema model not found in checkpoint, resetting to current model")
                model_ema = deepcopy(model_without_ddp)
            else:
                model_ema.load_state_dict(checkpoint["model_ema"])

    # 5단계: 평가 전용 모드

    def build_evaluator_list(base_ds, dataset_name):
        """Helper function to build the list of evaluators for a given dataset"""
        evaluator_list = []
        if args.no_detection:
            return evaluator_list
        iou_types = ["bbox"]
        if args.masks:
            iou_types.append("segm")

        evaluator_list.append(CocoEvaluator(base_ds, tuple(iou_types), useCats=False))
        if "refexp" in dataset_name:
            evaluator_list.append(RefExpEvaluator(base_ds, ("bbox")))
        if "clevrref" in dataset_name:
            evaluator_list.append(ClevrRefEvaluator(base_ds, ("bbox")))
        if "flickr" in dataset_name:
            evaluator_list.append(
                FlickrEvaluator(
                    args.flickr_dataset_path,
                    subset="test" if args.test else "val",
                    merge_boxes=args.GT_type == "merged",
                )
            )
        if "phrasecut" in dataset_name:
            evaluator_list.append(
                PhrasecutEvaluator(
                    "test" if args.test else "miniv",
                    ann_folder=args.phrasecut_orig_ann_path,
                    output_dir=os.path.join(output_dir, "phrasecut_eval"),
                    eval_mask=args.masks,
                )
            )
        return evaluator_list

    # --eval 설정 시 학습은 건너뛰고 평가만 수행
    # model.eval() 모드가 자동으로 켜짐 (gradient 계산 X)
    # Runs only evaluation, by default on the validation set unless --test is passed.
    if args.eval:
        test_stats = {}
        test_model = model_ema if model_ema is not None else model
        for i, item in enumerate(val_tuples):
            evaluator_list = build_evaluator_list(item.base_ds, item.dataset_name) # 데이터셋별로 다른 evaluator 객체 생성
            postprocessors = build_postprocessors(args, item.dataset_name) # 모델 출력 결과를 평가에 맞게 후처리하는 함수
            item = item._replace(evaluator_list=evaluator_list)
            print(f"Evaluating {item.dataset_name}")
            curr_test_stats = evaluate(
                model=test_model,
                criterion=criterion,
                contrastive_criterion=contrastive_criterion,
                qa_criterion=qa_criterion,
                postprocessors=postprocessors,
                weight_dict=weight_dict,
                data_loader=item.dataloader,
                evaluator_list=item.evaluator_list,
                device=device,
                args=args,
            ) # 평가 실행
            test_stats.update({item.dataset_name + "_" + k: v for k, v in curr_test_stats.items()}) # 전체 결과 저장

        log_stats = {
            **{f"test_{k}": v for k, v in test_stats.items()},
            "n_parameters": n_parameters,
        }
        print(log_stats)
        return

    # 6단계: 학습 루프

    # Runs training and evaluates after every --eval_skip epochs
    print("Start training")
    start_time = time.time()
    best_metric = 0.0
    # --start-epoch부터 --epochs까지 루프 돌며 학습 진행 (--resume 옵션이 있다면 중간부터 이어서 시작 가능)
    for epoch in range(args.start_epoch, args.epochs):
        if args.epoch_chunks > 0: # 데이터셋이 너무 크면 작은 조각으로 나눠 학습
            sampler_train = samplers_train[epoch % len(samplers_train)]
            data_loader_train = data_loaders_train[epoch % len(data_loaders_train)]
            print(f"Starting epoch {epoch // len(data_loaders_train)}, sub_epoch {epoch % len(data_loaders_train)}")
        else:
            print(f"Starting epoch {epoch}")
        if args.distributed: # 분산 학습 시, 샘플러에 현재 epoch 정보 전달
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            contrastive_criterion=contrastive_criterion,
            qa_criterion=qa_criterion,
            data_loader=data_loader_train,
            weight_dict=weight_dict,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            args=args,
            max_norm=args.clip_max_norm,
            model_ema=model_ema,
        ) # train_one_epoch 함수 호출
        # 매 epoch 마다 checkpoint.pth 저장
        if args.output_dir:
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            # learning rate drop 시점이나 2의 배수 에폭에서는 추가 저장
            # extra checkpoint before LR drop and every 2 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 2 == 0:
                checkpoint_paths.append(output_dir / f"checkpoint{epoch:04}.pth")
            # 현재 모델, optimizer, epoch 정보 등을 저장함
            for checkpoint_path in checkpoint_paths:
                dist.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "model_ema": model_ema.state_dict() if args.ema else None,
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "args": args,
                    },
                    checkpoint_path,
                )

        # 검증 (평가) 실행
        # --eval-skip 주기마다 val 데이터셋으로 평가 수행 (evaluate 함수 호출)
        if epoch % args.eval_skip == 0:
            test_stats = {}
            test_model = model_ema if model_ema is not None else model
            for i, item in enumerate(val_tuples):
                evaluator_list = build_evaluator_list(item.base_ds, item.dataset_name)
                item = item._replace(evaluator_list=evaluator_list)
                postprocessors = build_postprocessors(args, item.dataset_name)
                print(f"Evaluating {item.dataset_name}")
                curr_test_stats = evaluate(
                    model=test_model,
                    criterion=criterion,
                    contrastive_criterion=contrastive_criterion,
                    qa_criterion=qa_criterion,
                    postprocessors=postprocessors,
                    weight_dict=weight_dict,
                    data_loader=item.dataloader,
                    evaluator_list=item.evaluator_list,
                    device=device,
                    args=args,
                )
                test_stats.update({item.dataset_name + "_" + k: v for k, v in curr_test_stats.items()})
        else:
            test_stats = {}

        # 학습 결과(train_stats) + 평가 결과(test_stats) + 현재 epoch 기록
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if args.output_dir and dist.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if epoch % args.eval_skip == 0:
            if args.do_qa:
                metric = test_stats["gqa_accuracy_answer_total_unscaled"]
            else:
                metric = np.mean([v[1] for k, v in test_stats.items() if "coco_eval_bbox" in k])

            # 현재 성능이 지금까지 중 가장 좋다면 BEST_checkpoint.pth로 저장
            if args.output_dir and metric > best_metric:
                best_metric = metric
                checkpoint_paths = [output_dir / "BEST_checkpoint.pth"]
                # extra checkpoint before LR drop and every 100 epochs
                for checkpoint_path in checkpoint_paths:
                    dist.save_on_master(
                        {
                            "model": model_without_ddp.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch,
                            "args": args,
                        },
                        checkpoint_path,
                    )

    # 전체 학습 시간 측정
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__": # 스크립트가 직접 실행될 때만 아래 블록이 실행되도록 (import 하는 경우에는 실행 X)
    parser = argparse.ArgumentParser("DETR training and evaluation script", parents=[get_args_parser()]) # ArgumentParser 생성 (모든 커맨드라인 인자 정의 받아서 설정)
    args = parser.parse_args() # 커맨드라인에서 넘긴 인자들을 파싱해서 args라는 객체로 저장
    if args.output_dir: # --output-dir로 전달받은 경로가 존재하지 않으면 자동으로 폴더 생성
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args) # 방금 파싱한 args를 가지고 실제 학습/평가를 수행하는 main() 함수 호출
