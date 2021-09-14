# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import random
import time
from functools import partial
import sys

import numpy as np
import paddle
from paddle.io import DataLoader
from paddle.metric import Accuracy

from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import BertForSequenceClassification, BertTokenizer
from modeling import MobileBertForSequenceClassification
from model_tokenizer import MobileBertTokenizer

from utils import get_scheduler
from paddlenlp.metrics import AccuracyAndF1, Mcc, PearsonAndSpearman
import json
import math
from collections import OrderedDict

FORMAT = "%(asctime)s-%(levelname)s: %(message)s"
logger = logging.getLogger(__name__)


def save_json(data, file_path):
    with open(str(file_path), "w", encoding="utf8") as f:
        json.dump(data, f, ensure_ascii=False)


METRIC_CLASSES = {
    "cola": Mcc,
    "sst-2": Accuracy,
    "mrpc": AccuracyAndF1,
    "sts-b": PearsonAndSpearman,
    "qqp": AccuracyAndF1,
    "qnli": Accuracy,
    "mnli": Accuracy,
    "rte": Accuracy,
    "wnli": Accuracy,
}

MODEL_CLASSES = {
    "bert": (BertForSequenceClassification, BertTokenizer),
    "mobilebert": (MobileBertForSequenceClassification, MobileBertTokenizer),
}


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: "
        + ", ".join(METRIC_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_type",
        default="mobilebert",
        type=str,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default="mobilebert-uncased",
        type=str,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(
            sum(
                [
                    list(classes[-1].pretrained_init_configuration.keys())
                    for classes in MODEL_CLASSES.values()
                ],
                [],
            )
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--scheduler_type",
        default="cosine_annel",
        type=str,
        help="scheduler_type.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-4,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--layer_lr_decay", default=0.8, type=float, help="layer_lr_decay"
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--logging_steps", type=int, default=100, help="Log every X updates steps."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--weight_decay",
        default=0.01,
        type=float,
        help="Weight decay if we apply some.",
    )
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Linear warmup over warmup_steps. If > 0: Override warmup_proportion",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Linear warmup proportion over total steps.",
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="random seed for initialization"
    )
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        choices=["cpu", "gpu"],
        help="The device to select to train the model, is must be cpu/gpu.",
    )
    parser.add_argument(
        "--decay_radio", default=0.9, type=float, help="layer_lr_decay"
    )
    parser.add_argument(
        "--min_lr", default=5e-7, type=float, help="layer_lr_decay"
    )
    args = parser.parse_args()
    return args


def _get_layer_lr_radios(layer_decay=0.8, n_layers=12):
    """Have lower learning rates for layers closer to the input."""
    key_to_depths = OrderedDict(
        {
            "mpnet.embeddings.": 0,
            "mpnet.encoder.relative_attention_bias.": 0,
            "mpnet.pooler.": n_layers + 2,
            "mpnet.classifier.": n_layers + 2,
        }
    )
    for layer in range(n_layers):
        key_to_depths[f"mpnet.encoder.layer.{str(layer)}."] = layer + 1
    return {
        key: (layer_decay ** (n_layers + 2 - depth))
        for key, depth in key_to_depths.items()
    }


def set_seed(args):
    # Use the same data seed(for data shuffle) for all procs to guarantee data
    # consistency after sharding.
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Maybe different op seeds(for dropout) for different procs is better. By:
    # `paddle.seed(args.seed + paddle.distributed.get_rank())`
    paddle.seed(args.seed)


@paddle.no_grad()
def evaluate(model, loss_fct, metric, data_loader):
    model.eval()
    metric.reset()
    for batch in data_loader:
        input_ids, segment_ids, labels = batch
        logits = model(input_ids)
        loss = loss_fct(logits, labels)
        correct = metric.compute(logits, labels)
        metric.update(correct)
    res = metric.accumulate()
    if isinstance(metric, AccuracyAndF1):
        print(
            "eval loss: %f, acc: %s, precision: %s, recall: %s, f1: %s, acc and f1: %s, "
            % (
                loss.numpy(),
                res[0],
                res[1],
                res[2],
                res[3],
                res[4],
            )
        )
    elif isinstance(metric, Mcc):
        logger.info("eval loss: %f, mcc: %s, " % (loss.numpy(), res[0]))
    elif isinstance(metric, PearsonAndSpearman):
        logger.info(
            "eval loss: %f, pearson: %s, spearman: %s, pearson and spearman: %s, "
            % (loss.numpy(), res[0], res[1], res[2]),
            end="",
        )
    else:
        logger.info("eval loss: %f, acc: %s, " % (loss.numpy(), res))
    model.train()


def convert_example(example, tokenizer, label_list, max_seq_length=512, is_test=False):
    """convert a glue example into necessary features"""
    if not is_test:
        # `label_list == None` is for regression task
        label_dtype = "int64" if label_list else "float32"
        # Get the label
        label = example["labels"]
        label = np.array([label], dtype=label_dtype)
    # Convert raw text to feature
    if (int(is_test) + len(example)) == 2:
        example = tokenizer(example["sentence"], max_seq_len=max_seq_length)
    else:
        example = tokenizer(
            example["sentence1"],
            text_pair=example["sentence2"],
            max_seq_len=max_seq_length,
        )

    if not is_test:
        return example["input_ids"], example["token_type_ids"], label
    else:
        return example["input_ids"], example["token_type_ids"]


def do_train(args):
    paddle.set_device(args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                os.path.join(os.path.dirname(args.output_dir), "run.log"),
                mode="w",
                encoding="utf-8",
            ),
            logging.StreamHandler()
        ],
    )
    logger.info("**********  Configuration Arguments **********")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"{arg}: {value}")
    logger.info("**************************************************")


    args.task_name = args.task_name.lower()
    metric_class = METRIC_CLASSES[args.task_name]
    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    train_ds = load_dataset("glue", args.task_name, splits="train")
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        label_list=train_ds.label_list,
        max_seq_length=args.max_seq_length,
    )
    train_ds = train_ds.map(trans_func, lazy=True)
    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_ds, batch_size=args.batch_size, shuffle=True
    )
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
        Stack(dtype="int64" if train_ds.label_list else "float32"),  # label
    ): fn(samples)
    train_data_loader = DataLoader(
        dataset=train_ds,
        batch_sampler=train_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=True,
    )
    if args.task_name == "mnli":
        dev_ds_matched, dev_ds_mismatched = load_dataset(
            "glue", args.task_name, splits=["dev_matched", "dev_mismatched"]
        )

        dev_ds_matched = dev_ds_matched.map(trans_func, lazy=True)
        dev_ds_mismatched = dev_ds_mismatched.map(trans_func, lazy=True)
        dev_batch_sampler_matched = paddle.io.BatchSampler(
            dev_ds_matched, batch_size=args.batch_size * 2, shuffle=False
        )
        dev_data_loader_matched = DataLoader(
            dataset=dev_ds_matched,
            batch_sampler=dev_batch_sampler_matched,
            collate_fn=batchify_fn,
            num_workers=2,
            return_list=True,
        )
        dev_batch_sampler_mismatched = paddle.io.BatchSampler(
            dev_ds_mismatched, batch_size=args.batch_size * 2, shuffle=False
        )
        dev_data_loader_mismatched = DataLoader(
            dataset=dev_ds_mismatched,
            batch_sampler=dev_batch_sampler_mismatched,
            collate_fn=batchify_fn,
            num_workers=2,
            return_list=True,
        )
    else:
        dev_ds = load_dataset("glue", args.task_name, splits="dev")
        dev_ds = dev_ds.map(trans_func, lazy=True)
        dev_batch_sampler = paddle.io.BatchSampler(
            dev_ds, batch_size=args.batch_size * 2, shuffle=False
        )
        dev_data_loader = DataLoader(
            dataset=dev_ds,
            batch_sampler=dev_batch_sampler,
            collate_fn=batchify_fn,
            num_workers=2,
            return_list=True,
        )

    num_classes = 1 if train_ds.label_list == None else len(train_ds.label_list)
    print("num_class:",num_classes)
    
    model = model_class.from_pretrained('./weight/paddle')

    # reinit_encoder_layer_parameter(model, last_nums=4)

    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    # # layer_lr for base
    # ############################################################
    # if args.layer_lr_decay != 1.0:
    #     layer_lr_radios_map = _get_layer_lr_radios(args.layer_lr_decay, n_layers=12)
    #     for name, parameter in model.named_parameters():
    #         layer_lr_radio = 1.0
    #         for k, radio in layer_lr_radios_map.items():
    #             if k in name:
    #                 layer_lr_radio = radio
    #                 break
    #         parameter.optimize_attr["learning_rate"] *= layer_lr_radio
    # ############################################################

    num_training_steps = (
        args.max_steps
        if args.max_steps > 0
        else (len(train_data_loader) * args.num_train_epochs)
    )
    args.num_train_epochs = math.ceil(num_training_steps/len(train_data_loader))

    print('get lr_scheduler')
    lr_scheduler = get_scheduler(
        learning_rate=args.learning_rate,
        scheduler_type=args.scheduler_type,
        args=args,
        num_warmup_steps=args.warmup_steps
        if args.warmup_steps > 0
        else args.warmup_proportion,
        num_training_steps=num_training_steps,
    )
    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name
        for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "Norm"])
    ]

    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        beta1=0.9,
        beta2=0.98,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params,
    )

    loss_fct = (
        paddle.nn.loss.CrossEntropyLoss()
        if train_ds.label_list
        else paddle.nn.loss.MSELoss()
    )

    metric = metric_class()

    global_step = 0
    tic_train = time.time()

    print("num_training_steps", num_training_steps)
    # 6315
    # args.logging_steps = 20
    # args.save_steps = num_training_steps % 20
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    save_json(args.__dict__, os.path.join(args.output_dir, "args.json"))

    logger.info("********** Running training **********")
    logger.info(f"  Num examples = {len(train_data_loader.dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous train batch size = {args.batch_size}")
    logger.info(f"  Instantaneous eval batch size = {args.batch_size*2}")
    logger.info(f"  Total train batch size (w. accumulation) = {args.batch_size}")
    logger.info(f"  Total optimization steps = {num_training_steps}")
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_data_loader):
            global_step += 1

            input_ids, segment_ids, labels = batch
            logits = model(input_ids)
            # print(logits,labels)
            loss = loss_fct(logits, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()
            if (
                global_step % args.logging_steps == 0
                or global_step == num_training_steps
            ):
                logger.info(
                    "global step %d/%d, epoch: %d, batch: %d, rank_id: %s, loss: %f, lr: %.10f, speed: %.4f step/s"
                    % (
                        global_step,
                        num_training_steps,
                        epoch,
                        step,
                        paddle.distributed.get_rank(),
                        loss,
                        optimizer.get_lr(),
                        args.logging_steps / (time.time() - tic_train),
                    )
                )
                tic_train = time.time()
            if global_step % args.save_steps == 0 and global_step>5000 or global_step == num_training_steps:
                tic_eval = time.time()
                if args.task_name == "mnli":
                    logger.info("=" * 100)
                    logger.info("m_acc:")
                    evaluate(model, loss_fct, metric, dev_data_loader_matched)
                    logger.info("mm_acc:")
                    evaluate(model, loss_fct, metric, dev_data_loader_mismatched)
                    logger.info("eval done total : %s s" % (time.time() - tic_eval))
                    logger.info("=" * 100)
                else:
                    logger.info("=" * 100)
                    evaluate(model, loss_fct, metric, dev_data_loader)
                    logger.info("eval done total : %s s" % (time.time() - tic_eval))
                if paddle.distributed.get_rank() == 0 and global_step>5000:
                    output_dir = os.path.join(
                        args.output_dir,
                        "%s_ft_model_%d" % (args.task_name, global_step),
                    )
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Need better way to get inner model of DataParallel
                    model_to_save = (
                        model._layers
                        if isinstance(model, paddle.DataParallel)
                        else model
                    )
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
            if global_step >= num_training_steps:
                return


def print_arguments(args):
    """print arguments"""
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(vars(args).items()):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


if __name__ == "__main__":
    args = parse_args()
    if args.task_name.lower() in ["sts-b", "rte", "wnli"]:
        args.num_train_epochs = 10
    if args.output_dir is None:
        args.output_dir = args.task_name.lower()
    print_arguments(args)
    n_gpu = len(os.getenv("CUDA_VISIBLE_DEVICES", "").split(","))
    if args.device in "gpu" and n_gpu > 1:
        paddle.distributed.spawn(do_train, args=(args,), nprocs=n_gpu)
    else:
        do_train(args)