import sys
import json
import pickle
import random
import math
from collections import OrderedDict

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddlenlp.transformers import (
    CosineDecayWithWarmup,
    LinearDecayWithWarmup,
    PolyDecayWithWarmup,
)
from modeling import NoNorm

scheduler_type2cls = {
    "linear": LinearDecayWithWarmup,
    "cosine": CosineDecayWithWarmup,
    "poly": PolyDecayWithWarmup,
    "cosine_annel":None,
}
from paddle.optimizer.lr import LRScheduler
class CosineAnnealingWithWarmupDecay(LRScheduler):
    def __init__(self,
                 max_lr,
                 min_lr,
                 warmup_step,
                 decay_step,
                 last_epoch=0,
                 verbose=False):

        self.decay_step = decay_step
        self.warmup_step = warmup_step
        self.max_lr = max_lr
        self.min_lr = min_lr
        super(CosineAnnealingWithWarmupDecay, self).__init__(max_lr, last_epoch,
                                                             verbose)
    def get_lr(self):
        if self.warmup_step > 0 and self.last_epoch <= self.warmup_step:
            return float(self.max_lr) * (self.last_epoch) / self.warmup_step

        if self.last_epoch > self.decay_step:
            return self.min_lr

        num_step_ = self.last_epoch - self.warmup_step
        decay_step_ = self.decay_step - self.warmup_step
        decay_ratio = float(num_step_) / float(decay_step_)
        coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)
        return self.min_lr + coeff * (self.max_lr - self.min_lr)

def get_layer_lr_radios(layer_decay=0.8, n_layers=24):
    """Have lower learning rates for layers closer to the input."""
    key_to_depths = OrderedDict(
        {
            "mobilebert.embeddings.": 0,
            "qa_outputs.": n_layers + 2,
        }
    )
    for layer in range(n_layers):
        key_to_depths[f"mobilebert.encoder.layer.{str(layer)}."] = layer + 1
    return {
        key: (layer_decay ** (n_layers + 2 - depth))
        for key, depth in key_to_depths.items()
    }


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)

def is_integer(number):
    if sys.version > '3':
        return isinstance(number, int)
    return isinstance(number, (int, long))

def get_writer(args):
    if args.writer_type == "visualdl":
        from visualdl import LogWriter

        writer = LogWriter(logdir=args.logdir)
    elif args.writer_type == "tensorboard":
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(logdir=args.logdir)
    else:
        raise ValueError("writer_type must be in ['visualdl', 'tensorboard']")
    return writer


def get_scheduler(
    learning_rate,
    scheduler_type,
    args,
    num_warmup_steps=None,
    num_training_steps=None,
    **scheduler_kwargs,
):
    if scheduler_type not in scheduler_type2cls.keys():
        data = " ".join(scheduler_type2cls.keys())
        raise ValueError(f"scheduler_type must be choson from {data}")

    if num_warmup_steps is None:
        raise ValueError(f"requires `num_warmup_steps`, please provide that argument.")

    if num_training_steps is None:
        raise ValueError(
            f"requires `num_training_steps`, please provide that argument."
        )
    print("scheduler_type:",scheduler_type)
    if scheduler_type=='cosine_annel':
        num_warmup_steps = num_warmup_steps if is_integer(num_warmup_steps) else int(
            math.floor(num_warmup_steps * num_training_steps))
        print("num_warmup_steps:", num_warmup_steps)
        return CosineAnnealingWithWarmupDecay(max_lr=learning_rate, min_lr=args.min_lr, warmup_step=num_warmup_steps, decay_step=math.ceil(num_training_steps*args.decay_radio))
    else:

        return scheduler_type2cls[scheduler_type](
            learning_rate=learning_rate,
            total_steps=num_training_steps,
            warmup=num_warmup_steps,
            **scheduler_kwargs,
        )


def save_json(data, file_name):
    with open(file_name, "w", encoding="utf-8") as w:
        w.write(json.dumps(data, ensure_ascii=False, indent=4) + "\n")

def reinit_encoder_layer_parameter(model, last_nums=4):
    print("reinit_encoder_layer_parameter layer num:{}".format(last_nums))
    encoder_layers=model.mobilebert.encoder.layer
    encoder_layer_num=len(encoder_layers)
    print(encoder_layer_num)
    begin_init_layer=encoder_layer_num-last_nums
    for i in range(last_nums):
        layer = encoder_layers[begin_init_layer+i]
        for m in layer.sublayers():
            print(i)
            if isinstance(m, nn.Linear):
                print("linear.",m.weight.shape)
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
                m.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=0.02,
                        shape=m.weight.shape)
                )
                # m.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if m.bias is not None:
                    m.bias.set_value(paddle.zeros_like(m.bias))
            elif isinstance(m, (nn.LayerNorm, NoNorm)):
                print("NoNorm.",m.weight.shape)
                m.bias.set_value(paddle.zeros_like(m.bias))
                m.weight.set_value(paddle.ones_like(m.weight))
    

class CrossEntropyLossForSQuAD(nn.Layer):
    def forward(self, logits, labels):
        start_logits, end_logits = logits
        start_position, end_position = labels
        ignored_index = start_logits.shape[-1]
        # print(start_position.numpy())
        if np.any(start_position.numpy()>ignored_index) or np.any(start_position.numpy()<0):
            print('start_position outuput range, start_logits shape is {}'.format(start_logits))
        
        if np.any(end_position.numpy()>ignored_index) or np.any(end_position.numpy()<0):
            print('end_position outuput range, start_logits shape is {}'.format(start_logits))


        start_position = paddle.unsqueeze(start_position, axis=-1)
        end_position = paddle.unsqueeze(end_position, axis=-1)
        # ignored_index = start_logits.shape
        # print(start_logits.shape)
        
        start_loss = F.cross_entropy(input=start_logits, label=start_position)
        end_loss = F.cross_entropy(input=end_logits, label=end_position)
        loss = (start_loss + end_loss) / 2

        return loss


def save_pickle(data, file_path):
    with open(str(file_path), "wb") as f:
        pickle.dump(data, f)


def load_pickle(input_file):
    with open(str(input_file), "rb") as f:
        data = pickle.load(f)
    return data