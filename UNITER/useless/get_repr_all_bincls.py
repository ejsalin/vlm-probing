import random
import argparse
import json
import os
from os.path import exists
from time import time
import pickle 
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import re
import numpy as np
from apex import amp
from horovod import torch as hvd



from collections import defaultdict

from pytorch_pretrained_bert import BertTokenizer

from data import (DetectFeatLmdb, TxtTokLmdb,
                  PrefetchLoader, TokenBucketSampler,
                  DetectFeatTxtTokDataset,rpr_collate,rpr_collate_dec,RPRDataset,RPRDataset_decorrelated)
from model.model import UniterConfig  , UniterModel
from model.nlvr2 import UniterForNlvr2Paired
from model.vqa import UniterForVisualQuestionAnswering
from model.pretrain import UniterForPretraining

from utils.misc import Struct
from utils.const import IMG_DIM, BUCKET_SIZE

def tokenToString(text):
  tokens = [token for token in text.split(" ") if token not in  ["[CLS]","[SEP]","[PAD]"]]
  toks =[]
  for i,t in enumerate(tokens):
    if t.startswith("##"):
      if toks != []:
        toks[-1] = toks[-1]+t[2:].lower()
    else:
      toks.append(t.lower())
  return " ".join(toks).lower()

def shift(tensor,words,inv,t):

    if t >0.5:
        #shift
        tensor_list = tensor.tolist() 

        both = set(tensor_list).intersection(words)
        tensor_list_A = [tensor_list.index(x) for x in both]
        #print(both)
        #print(words)
        #print(tensor_list)
        for j,i in enumerate(tensor_list_A) :

            c= inv[list(both)[j]]

            tensor_list[i] = c
        return torch.tensor(tensor_list),1,both
    else:
        return tensor ,0,[]

def main(opts):

    if opts.task=="size":
        words ={"large":"small","small":"tall","little":"big","big":"little","tall":"small","long":"short","short":"long"}
    if opts.task=="position":
        words ={"top":"bottom","bottom":"top","front":"back","back":"front","under":"above","above":"under","beside":"beyond","beyond":"beside"}


    print(opts.model,' dec ?',opts.dec,' img_data :',opts.img_db )
    hvd.init()

    device = torch.device("cuda")  
    
    
    train_opts = Struct(json.load(open('config/pretrain-alldata-base-8gpu.json')))

    
    img_db = DetectFeatLmdb(opts.img_db,
                            train_opts.conf_th, train_opts.max_bb,
                            train_opts.min_bb, train_opts.num_bb,
                            opts.compressed_db)                   
    txt_db = TxtTokLmdb(opts.txt_db, -1)
   

    if opts.dec != 1:

        da = RPRDataset
        rprfnc =rpr_collate
    else :
        print("decorelated")
        da = RPRDataset_decorrelated
        rprfnc =rpr_collate_dec
    dset = da(txt_db, img_db)
    print(f'RPRDataset len :{len(dset.ids)}')

    batch_size = (train_opts.val_batch_size if opts.batch_size is None
                  else opts.batch_size)
                  
    print(f'batch_size,{batch_size}')

    eval_dataloader = DataLoader(dset, batch_size=batch_size,
                                 num_workers=1,
                                 pin_memory=opts.pin_mem,
                                 collate_fn=rprfnc)
    print(f'eval DataLoader len :{len(eval_dataloader.dataset)}')
    eval_dataloader = PrefetchLoader(eval_dataloader)

    print(f'eval PrefetchLoader len :{len(eval_dataloader.dataset)}')

    # Prepare model
    if opts.model == 'base':
        config_path="config/uniter-base.json"
        checkpoint = torch.load("pretrained/uniter-base.pt")
        IMG_DIM=2048
        IMG_LABEL_DIM=1601
        model = UniterForPretraining.from_pretrained(
            config_path, checkpoint,
            img_dim=IMG_DIM, img_label_dim=IMG_LABEL_DIM)
        model.to(device)
        print("uniter base loaded")

    if opts.model == 'nlvr':
        
        checkpoint = torch.load("pretrained/nlvr-base/ckpt/model_step_6500.pt")
        IMG_DIM=2048
        #IMG_LABEL_DIM=1601
        model_config = UniterConfig.from_json_file('pretrained/nlvr-base/log/model.json')
        model = UniterForNlvr2Paired(model_config, img_dim=IMG_DIM)
        model.init_type_embedding()
        model.load_state_dict(checkpoint, strict=False)
        model.to(device)
        print("NLVR2 loaded")
        
    if opts.model == 'vqa':
        ans2label_file = 'pretrained/VQA/ckpt/ans2label.json'
        ans2label = json.load(open(ans2label_file))
        config_path="pretrained/VQA/log/model.json"
        checkpoint = torch.load("pretrained/VQA/ckpt/model_step_6000.pt")
        IMG_DIM=2048
        #IMG_LABEL_DIM=1601
        model = UniterForVisualQuestionAnswering.from_pretrained(
            config_path, checkpoint,
            img_dim=IMG_DIM,num_answer=len(ans2label))
        model.to(device)
        print("VQA loaded")

    evaluate(model, eval_dataloader, device,opts,words)
   

def get_id(fname):
    r1 = re.search(r'0[1-9][0-9]+.npz',fname)
    return r1.group(0)[1:-4]


@torch.no_grad()
def evaluate(model, eval_loader, device,opts,dict):
    filename = "../data_used/"+opts.task+"_"+opts.data+".pickle" 
    with open(filename, 'rb') as handle:
        data_ = pickle.load(handle)
    #data_ = {}
    print("start running evaluation...")
    toker = BertTokenizer.from_pretrained("bert-base-cased")
    inv = {} 
    words = list(dict.keys())

    for a,b in dict.items():
        inv[toker.convert_tokens_to_ids([a])[0]] = toker.convert_tokens_to_ids([b])[0] 
    words_ = [toker.convert_tokens_to_ids([a])[0] for a in list(dict.keys())]
    print(words)
    print(words_)
    print(inv)

    model.eval()
    n_ex = 0
    st = time()
    results = []
    t0 = time()
    texts =[]
    fnames =[]
    tensor_out = []
    tensor_out_pooled=[]
    text_ids=[]
    gather_ids=[]
    targets = []


    for  batch in tqdm(eval_loader,total =len(eval_loader.dataset)):
        

        #########################################################################
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attention_mask = batch['attn_masks']
        gather_index = batch['gather_index']
        t = " ".join(toker.convert_ids_to_tokens(batch['input_ids'][0].tolist()))

        if  len(set.intersection(set(t.split(" ")),set(words))): 
            a = data_[tokenToString(t)]
            #a = random.random()
            input_idss,target,shifted_id = shift(input_ids.squeeze(0),words_,inv,a[1])
            

            with torch.no_grad():

                output = model.uniter(input_idss.unsqueeze(0).to(device), position_ids,
                                          img_feat, img_pos_feat,
                                          attention_mask, gather_index,
                                          output_all_encoded_layers=False)
                output_pooled= model.uniter.pooler(output)
            #print()
            #print(tokenToString(t))
            changed_text =tokenToString(" ".join(toker.convert_ids_to_tokens(input_idss.tolist())))
            #print(changed_text)
            #print(target)
            #print()
            data_[tokenToString(t)] =  [changed_text,target,shifted_id,get_id(batch['im_fnames'][0])]
            texts.append(t)
            fnames.append(batch['im_fnames'][0])
            tensor_out.append(output.to('cpu'))
            tensor_out_pooled.append(output_pooled.to('cpu'))
            text_ids.append(position_ids)
            gather_ids.append(gather_index)

            targets.append(target) 

            del output_pooled
            del output
            torch.cuda.empty_cache()

    #filename = "../representations/"+opts.task+"_"+opts.data+".pickle" 
    #with open(filename, 'wb') as handle:
    #    pickle.dump(data_, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    print(len(texts),len(fnames))
    repre = {"fnames":fnames , "texts":texts ,'representations':tensor_out,'pooled_representation':tensor_out_pooled,
            "text_ids":text_ids ,"gather_ids":gather_ids,"targets":targets }
    print(np.mean(targets))
    filename = "../representations/uniter_bincls_"+opts.task+"_"+opts.model+"_"+opts.data+".pickle" if opts.dec==0 else "../representations/uniter_bincls_"+opts.task+"_"+opts.model+"_"+opts.data+"_dec.pickle"
    with open(filename, 'wb') as handle:
        pickle.dump(repre, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(filename +" saved")
  


   


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters

    parser.add_argument("--dec",
                        type=int, choices=[0,1],required=True,
                        help="The input train corpus.")
    parser.add_argument("--task",
                        type=str, choices=["position","size"],required=True,
                        help="The input train corpus.")
    parser.add_argument("--txt_db",
                        type=str, required=True,
                        help="The input train corpus.")
    parser.add_argument("--model",
                        type=str, required=True,choices=['base', 'nlvr', 'vqa'],
                        help="The input train corpus.")
    parser.add_argument("--img_db",
                        type=str, required=True,
                        help="The input train images.")
    parser.add_argument("--data",
                        type=str, required=True,choices=['coco', 'flickr'],
                        help="data_used file.")

    parser.add_argument('--compressed_db', action='store_true',
                        help='use compressed LMDB')
    parser.add_argument("--batch_size", type=int,default=1,
                        help="batch size for evaluation")
    parser.add_argument('--n_workers', type=int, default=1,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true',
                        help="pin memory")
    #parser.add_argument('--fp16', action='store_true',
    #                   help="fp16 inference")

    #parser.add_argument("--train_dir", type=str, required=True,
    #                   help="The directory storing NLVR2 finetuning output")
    #parser.add_argument("--ckpt", type=int, required=True,
    #                    help="specify the checkpoint to run inference")
    #parser.add_argument("--output_dir", type=str, required=True,
     #                   help="The output directory where the prediction "
     #                        "results will be written.")
    args = parser.parse_args()

    main(args)
