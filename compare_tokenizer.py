import torch
import paddle

from transformers import MobileBertTokenizerFast as MobileBertTokenizerPt

from model_tokenizer import MobileBertTokenizerV2 as MobileBertTokenizerV2Pd
import numpy as np


def compare_detail(data1, data2):
        return np.any((np.array(data1)-np.array(data2))==0)

def compare_tokenizer(token_pd, token_pt):
    for i in range(2):
        print("token {}:".format(i))
        input_ids_pd=token_pd[i]['input_ids']
        input_ids_tfm=token_pt['input_ids'][i]
        if compare_detail(input_ids_pd,input_ids_tfm):
            print("\t input_ids aligned")
        token_type_ids_pd=token_pd[i]['token_type_ids']
        token_type_ids_tfm=token_pt['token_type_ids'][i]
        if compare_detail(token_type_ids_pd,token_type_ids_tfm):
            print("\t token_type_ids aligned")
        offset_mapping_pd=token_pd[i]['offset_mapping']
        offset_mapping_tfm=token_pt['offset_mapping'][i]
        assert len(offset_mapping_pd)==len(offset_mapping_tfm)
        off_set_align=[offset_mapping_pd[i]==offset_mapping_tfm[i] for i in range(len(offset_mapping_tfm))]
        if np.any(np.array(off_set_align)):
            print("\t offset_mapping aligned")

        if token_pd[i]['overflow_to_sample']==token_pt['overflow_to_sample_mapping'][i]:
            print('\t overflow_to_sample aligned.')

if __name__ == "__main__":
    tiny_text = "Who is the head coach of the Broncos?"
    context_text = "Following their loss in the divisional round of the previous season's playoffs, the Denver Broncos underwent numerous coaching changes, including a mutual parting with head coach John Fox (who had won four divisional championships in his four years as Broncos head coach), and the hiring of Gary Kubiak as the new head coach. Under Kubiak, the Broncos planned to install a run-oriented offense with zone blocking to blend in with quarterback Peyton Manning's shotgun passing skills, but struggled with numerous changes and injuries to the offensive line, as well as Manning having his worst statistical season since his rookie year with the Indianapolis Colts in 1998, due to a plantar fasciitis injury in his heel that he had suffered since the summer, and the simple fact that Manning was getting old, as he turned 39 in the 2015 off-season. Although the team had a 7–0 start, Manning led the NFL in interceptions. In week 10, Manning suffered a partial tear of the plantar fasciitis in his left foot. He set the NFL's all-time record for career passing yards in this game, but was benched after throwing four interceptions in favor of backup quarterback Brock Osweiler, who took over as the starter for most of the remainder of the regular season. Osweiler was injured, however, leading to Manning's return during the Week 17 regular season finale, where the Broncos were losing 13–7 against the 4–11 San Diego Chargers, resulting in Manning re-claiming the starting quarterback position for the playoffs by leading the team to a key 27–20 win that enabled the team to clinch the number one overall AFC seed. Under defensive coordinator Wade Phillips, the Broncos' defense ranked number one in total yards allowed, passing yards allowed and sacks, and like the previous three seasons, the team has continued to set numerous individual, league and franchise records. With the defense carrying the team despite the issues with the offense, the Broncos finished the regular season with a 12–4 record and earned home-field advantage throughout the AFC playoffs."
    mobert_tokenizer=MobileBertTokenizerPt.from_pretrained('google/mobilebert-uncased')
    mobert_tokenizerv2 = MobileBertTokenizerV2Pd.from_pretrained('mobilebert-uncased')

    token_pt = mobert_tokenizer(
        [tiny_text],[context_text],
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=False,
        return_attention_mask=False,
        return_token_type_ids=True,
    )
    token_pd = mobert_tokenizerv2([tiny_text],[context_text],stride=128,max_seq_len=384)
    
    compare_tokenizer(token_pd, token_pt)