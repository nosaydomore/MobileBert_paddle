import paddle

from args import parse_args
from data import get_dev_dataloader
from train_squad import MODEL_CLASSES, evaluate
import os


def main(args):
    paddle.set_device(args.device)
    model_class, tokenizer_class, args.need_token_type_ids = MODEL_CLASSES[
        args.model_type
    ]
    
    if args.use_huggingface_tokenizer and args.model_type == "mobilebert":
        from transformers import MobileBertTokenizerFast
        tokenizer = MobileBertTokenizerFast.from_pretrained("google/mobilebert-uncased")
    else:
        print("paddle tokenizer.")
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    dev_data_loader = get_dev_dataloader(tokenizer, args)
    
    if not args.version_2_with_negative:
        model = model_class.from_pretrained("./task/squadv1/step-6720")
    else:
        model = model_class.from_pretrained("./task/squadv2/step-10320")

    evaluate(model, dev_data_loader, args, output_dir="./")


if __name__ == "__main__":
    args = parse_args()
    main(args)