import logging
import math
import os
import sys
sys.path.append('/home/aistudio/external-libraries')

import paddle
from paddle.amp import GradScaler, auto_cast
from paddle.optimizer import AdamW
from paddlenlp.transformers import (
    BertForQuestionAnswering,
    BertTokenizer
)
from tqdm import tqdm

from args import parse_args
from data import get_dev_dataloader, get_train_dataloader
from modeling import MobileBertForQuestionAnswering
from model_tokenizer import MobileBertTokenizerV2
from metric import compute_prediction, squad_evaluate
from utils import (
    CrossEntropyLossForSQuAD,
    get_scheduler,
    get_writer,
    save_json,
    set_seed,
    reinit_encoder_layer_parameter,
    get_layer_lr_radios,
)

logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    "bert": (BertForQuestionAnswering, BertTokenizer, True),
    "mobilebert":(MobileBertForQuestionAnswering, MobileBertTokenizerV2, True)
}


@paddle.no_grad()
def evaluate(model, data_loader, args, output_dir="./"):
    model.eval()
    all_start_logits = []
    all_end_logits = []

    for batch in data_loader:
        input_ids, token_type_ids = batch
        start_logits_tensor, end_logits_tensor = (
            model(input_ids, token_type_ids=token_type_ids)
            if args.need_token_type_ids
            else model(input_ids)
        )
        all_start_logits.extend(start_logits_tensor.numpy().tolist())
        all_end_logits.extend(end_logits_tensor.numpy().tolist())

    all_predictions, all_nbest_json, scores_diff_json = compute_prediction(
        data_loader.dataset.data,
        data_loader.dataset.new_data,
        (all_start_logits, all_end_logits),
        args.version_2_with_negative,
        args.n_best_size,
        args.max_answer_length,
        args.null_score_diff_threshold,
    )

    save_json(all_predictions, os.path.join(output_dir, "all_predictions.json"))
    if args.save_nbest_json:
        save_json(all_nbest_json, os.path.join(output_dir, "all_nbest_json.json"))

    eval_results = squad_evaluate(
        examples=data_loader.dataset.data,
        preds=all_predictions,
        na_probs=scores_diff_json,
    )
    return eval_results


def main(args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                os.path.join(os.path.dirname(args.output_dir), "run.log"),
                mode="w",
                encoding="utf-8",
            )
        ],
    )
    logger.info("**********  Configuration Arguments **********")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"{arg}: {value}")
    logger.info("**************************************************")
    paddle.set_device(args.device)
    set_seed(args)
    writer = get_writer(args)

    # get model and tokenizer
    model_class, tokenizer_class, args.need_token_type_ids = MODEL_CLASSES[
        args.model_type
    ]
    # model = model_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained('./weight/paddle')
    # paddle_checkpoint_path='./outputs_squadv1/init_param/model_state.pdparams'
    # mob_bert_pd_state_dict=paddle.load(paddle_checkpoint_path)
    print("loading pretrained model...")
    # model.load_dict(mob_bert_pd_state_dict)
    # reinit_encoder_layer_parameter(model, last_nums=4)
    model_init_path = os.path.join(args.output_dir,"init_param")
    model.save_pretrained(model_init_path)

    #======================================================
    # if args.layer_lr_decay != 1.0:
    #     layer_lr_radios_map = get_layer_lr_radios(args.layer_lr_decay, n_layers=24)
    #     for name, parameter in model.named_parameters():
    #         layer_lr_radio = 1.0
    #         for k, radio in layer_lr_radios_map.items():
    #             if k in name:
    #                 print(k,radio)
    #                 layer_lr_radio = radio
    #                 break
    #         parameter.optimize_attr["learning_rate"] *= layer_lr_radio
    #======================================================
    if args.use_huggingface_tokenizer and args.model_type == "mobilebert":
        from transformers import MobileBertTokenizerFast as MobileBertTokenizerPt
        tokenizer = MobileBertTokenizerPt.from_pretrained("./data_mobert")
    else:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    # get dataloader
    train_dataloader = get_train_dataloader(tokenizer, args)
    dev_dataloader = get_dev_dataloader(tokenizer, args)
    
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps > 0:
        args.num_train_epochs = math.ceil(
            args.max_train_steps / num_update_steps_per_epoch
        )
    else:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # get lr_scheduler
    print('get lr_scheduler')
    lr_scheduler = get_scheduler(
        learning_rate=args.learning_rate,
        scheduler_type=args.scheduler_type,
        args=args,
        num_warmup_steps=args.warmup_steps
        if args.warmup_steps > 0
        else args.warmup_radio,
        num_training_steps=args.max_train_steps,
    )

    total_batch_size = args.train_batch_size * args.gradient_accumulation_steps

    decay_params = [
        p.name
        for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "Norm"])
    ]
    optimizer = AdamW(
        learning_rate=lr_scheduler,
        beta1=0.9,
        beta2=0.98,
        # beta2=0.999,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params,
    )

    loss_fn = CrossEntropyLossForSQuAD()

    if args.use_amp:
        scaler = GradScaler(init_loss_scaling=args.scale_loss)

    logger.info("********** Running training **********")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous train batch size = {args.train_batch_size}")
    logger.info(f"  Instantaneous eval batch size = {args.eval_batch_size}")
    logger.info(f"  Total train batch size (w. accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    save_json(vars(args), os.path.join(args.output_dir, "args.json"))
    progress_bar = tqdm(range(args.max_train_steps))
    if args.version_2_with_negative:
        begin_save_step=5000
    else:
        begin_save_step=4000
    global_steps = 0
    tr_loss, logging_loss = 0.0, 0.0
    print('start train model.')
    for _ in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            model.train()
            with auto_cast(
                args.use_amp, custom_white_list=["layer_norm", "softmax", "gelu"]
            ):
                input_ids, token_type_ids, start_positions, end_positions = batch
                logits = (
                    model(input_ids, token_type_ids=token_type_ids)
                    if args.need_token_type_ids
                    else model(input_ids)
                )
                loss = (
                    loss_fn(logits[:2], (start_positions, end_positions))
                    / args.gradient_accumulation_steps
                )
                tr_loss += loss.item()

            if args.use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (
                step % args.gradient_accumulation_steps == 0
                or step == len(train_dataloader) - 1
            ):
                if args.use_amp:
                    scaler.minimize(optimizer, loss)
                else:
                    optimizer.step()

                lr_scheduler.step()
                optimizer.clear_grad()
                progress_bar.update(1)
                global_steps += 1
                if args.logging_steps > 0 and global_steps % args.logging_steps == 0:
                    writer.add_scalar("lr", lr_scheduler.get_lr(), global_steps)
                    writer.add_scalar(
                        "loss",
                        (tr_loss - logging_loss) / args.logging_steps,
                        global_steps,
                    )
                    logger.info(
                        "global_steps {} - lr: {:.10f}  loss: {:.8f}".format(
                            global_steps,
                            lr_scheduler.get_lr(),
                            (tr_loss - logging_loss) / args.logging_steps,
                        )
                    )
                    tmp_loss=(tr_loss - logging_loss) / args.logging_steps
                    logging_loss = tr_loss

                # if args.save_steps > 0 and global_steps % args.save_steps == 0 and global_steps>=10000:
                if args.save_steps > 0 and global_steps % args.save_steps == 0 and global_steps>=begin_save_step and tmp_loss<1.0:
                    logger.info("********** Running evaluating **********")
                    logger.info(f"********** Step {global_steps} **********")
                    output_dir = os.path.join(args.output_dir, f"step-{global_steps}")
                    os.makedirs(output_dir, exist_ok=True)
                    eval_results = evaluate(model, dev_dataloader, args, output_dir)
                    for k, v in eval_results.items():
                        if "exact" in k or "f1" in k:
                            writer.add_scalar(f"eval/{k}", v, global_steps)
                        logger.info(f"  {k} = {v}")
                    if global_steps>=begin_save_step:
                        model.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                    logger.info("********** Evaluating Done **********")

            if global_steps >= args.max_train_steps:
                logger.info("********** Training Done **********")
                return


if __name__ == "__main__":
    args = parse_args()
    main(args)