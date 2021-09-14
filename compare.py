import torch
import paddle

from modeling import MobileBertForSequenceClassification as MobileBertForSequenceClassificationPd
from transformers import MobileBertForSequenceClassification as MobileBertForSequenceClassificationPt, MobileBertTokenizerFast as MobileBertTokenizerPt

from model_tokenizer import MobileBertTokenizer as MobileBertTokenizerPd, MobileBertTokenizerV2 as MobileBertTokenizerV2Pd
import numpy as np


def compare_output(feat_pd, feat_pt):
    feat_pd=torch.tensor(feat_pd.numpy())
    diff = (feat_pd-feat_pt).abs()
    diff_mean = diff.mean()
    diff_max = diff.max()
    print("diff_mean:", diff_mean)
    print("diff_max:", diff_max)

if __name__ == "__main__":
    tiny_text = "Who is the head coach of the Broncos?"
    mobert_tokenizerv2 = MobileBertTokenizerV2Pd.from_pretrained('mobilebert-uncased')
    
    token_pd = mobert_tokenizerv2([tiny_text],max_seq_len=384)
    
    cls_model_pd = MobileBertForSequenceClassificationPd.from_pretrained("./weight/paddle")
    cls_model_pt = MobileBertForSequenceClassificationPt.from_pretrained("./weight/torch")

    cls_model_pd.eval()
    cls_model_pt.eval()

    input_ids_pd = paddle.to_tensor(token_pd[0]["input_ids"]).unsqueeze(0)
    token_type_ids_pd = paddle.to_tensor(token_pd[0]["token_type_ids"]).unsqueeze(0)

    input_ids_pt = torch.tensor(token_pd[0]["input_ids"]).unsqueeze(0)
    token_type_ids_pt = torch.tensor(token_pd[0]["token_type_ids"]).unsqueeze(0)

    with paddle.no_grad():
        output_pd=cls_model_pd(input_ids_pd,do_compare=True)
    with torch.no_grad():
        output_pt=cls_model_pt(input_ids_pt,output_hidden_states=True)

    sent_feat_pd=output_pd[1]
    sent_feat_pt=output_pt.hidden_states[-1]
    
    cls_feat_pd=sent_feat_pd[0][0]
    cls_feat_pt=sent_feat_pt[0][0]
    print("sent_feat_pd:",sent_feat_pd.shape)
    print("sent_feat_pt:",sent_feat_pt.shape)
    print("cls_feat_pd:",cls_feat_pd.shape)
    print("cls_feat_pt:",cls_feat_pt.shape)
    print("cls token feat:")
    compare_output(cls_feat_pd,cls_feat_pt)

    print("other token feat:")
    compare_output(sent_feat_pd[0][1:],sent_feat_pt[0][1:])
    print("sep token feat:")
    compare_output(sent_feat_pd[0][10],sent_feat_pt[0][10])
