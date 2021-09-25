import torch
from collections import OrderedDict
import paddle

def convert_pytorch_dict_to_paddle(torch_model_path,paddle_model_path):
    torch_model_dict=torch.load(torch_model_path, map_location="cpu")
    paddle_model_dict=OrderedDict()
    trans=['embedding_transformation.weight','query.weight','key.weight',
            'dense.weight','value.weight','decoder.weight','seq_relationship.weight']
    trans_no=0
    for k,v in torch_model_dict.items():
        if 'cls' in k:
            continue
        if any([t in k for t in trans]):
            print("transpose paramter of key:{}".format(k))
            trans_no+=1
            v=v.transpose(0,1)
        paddle_model_dict[k]=paddle.to_tensor(v.data.numpy())
    print("transpose parameter {}".format(trans_no))
    print("finnish convert pytorch to paddle.")
    paddle.save(paddle_model_dict,paddle_model_path)
    print("model_saved: {}".format(paddle_model_path))

    

if __name__ == "__main__":
    torch_checkpoint_path='./weight/torch/pytorch_model.bin'
    paddle_checkpoint_path="./weight/paddle/model_state.pdparams"
    convert_pytorch_dict_to_paddle(torch_checkpoint_path, paddle_checkpoint_path)


