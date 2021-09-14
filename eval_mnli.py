from train_glue import METRIC_CLASSES, MODEL_CLASSES, parse_args, convert_example
from paddlenlp.datasets import load_dataset
import paddle
from paddle.io import DataLoader
from paddlenlp.data import Stack, Tuple, Pad
from functools import partial


@paddle.no_grad()
def evaluate(model, metric, data_loader):
    model.eval()
    metric.reset()
    for batch in data_loader:
        input_ids, segment_ids, labels = batch
        logits = model(input_ids)
        # loss = loss_fct(logits, labels)
        correct = metric.compute(logits, labels)
        metric.update(correct)
    res = metric.accumulate()
    
    print("eval acc: %s, " % (res))


def get_mnli_dataloader(args, tokenizer):
    dev_ds_matched, dev_ds_mismatched = load_dataset(
        "glue", args.task_name, splits=["dev_matched", "dev_mismatched"]
    )
    train_ds = load_dataset("glue", args.task_name, splits="train")

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        label_list=train_ds.label_list,
        max_seq_length=args.max_seq_length,
    )
    dev_ds_matched = dev_ds_matched.map(trans_func, lazy=True)
    dev_ds_mismatched = dev_ds_mismatched.map(trans_func, lazy=True)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
        Stack(dtype="int64" if train_ds.label_list else "float32"),  # label
    ): fn(samples)

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
    return dev_data_loader_matched,dev_data_loader_mismatched

def do_eval():
    args = parse_args()
    paddle.set_device(args.device)
    args.task_name = args.task_name.lower()
    metric_class = METRIC_CLASSES[args.task_name]
    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    paddle_checkpoint='task/mnli/mnli_ft_model_17040'
    model = model_class.from_pretrained(paddle_checkpoint)
    metric = metric_class()

    dev_data_loader_matched, dev_data_loader_mismatched=get_mnli_dataloader(args,tokenizer)
    print('m acc:')
    evaluate(model, metric, dev_data_loader_matched)
    print('mm acc:')
    evaluate(model, metric, dev_data_loader_mismatched)

if __name__ == '__main__':
    do_eval()

    
    






