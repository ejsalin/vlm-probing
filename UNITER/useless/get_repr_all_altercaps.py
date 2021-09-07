#horovodrun -np 1 python get_representation.py --img_db ../DATA/flickr30k/ --txt_db ../DATA/flickr_txt_db/ --batch_size 1
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
from apex import amp
from horovod import torch as hvd

from cytoolz import curry


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

from utils.misc import Struct
from utils.const import IMG_DIM, BUCKET_SIZE
@curry
def bert_tokenize(tokenizer, text):
    ids = []
    for word in text.strip().split():
        ws = tokenizer.tokenize(word)
        if not ws:
            # some special char
            continue
        ids.extend(tokenizer.convert_tokens_to_ids(ws))
    return ids

def tokenToString(text):
  tokens = [token for token in text.split(" ") if token not in  ["[CLS]","[SEP]","[PAD]","."]]
  toks =[]
  for i,t in enumerate(tokens):
    if t.startswith("##"):
      toks[-1] = toks[-1]+t[2:].lower()
    else:
      toks.append(t.lower())

  res = " ".join(toks).lower()
  while res[-1] == ' ':
      res = res[:-1]
  return res

def get_url_id(fname):
    r1 = re.search(r'0[1-9][0-9]+.jpg',fname)
    return r1.group(0)[1:-4]

def get_id(fname):
    r1 = re.search(r'0[1-9][0-9]+.npz',fname)
    return r1.group(0)[1:-4]

with open('../DATA/pos_captions_subset.json') as json_file:
    data = json.load(json_file)
added_files = []
caps = []
non_cahnged_captions = [ tokenToString(data[k]['caption']) for k in data.keys() ]
captions = [(data[k]['caption'],data[k]['alternative_caption'],data[k]['replacement'],data[k]['image_url'])  for k in data.keys() ]
print("captions: " ,len(captions))
print("non_cahnged_captions: " ,len(set(non_cahnged_captions)))


print(data.keys())



def make_change(inputt,toker,data,device):
    tokenizer = bert_tokenize(toker)
    t =tokenToString(" ".join(toker.convert_ids_to_tokens(inputt.tolist())))
    changed = False
    target = 0
    results= []
    for caption,alter_cap,rep,_ in data:
        if tokenToString(caption) == t and alter_cap not in caps:
            print("text : ",t)

            print("alter : ",alter_cap)
            try :
                print("remplacement : ",rep,[toker.convert_tokens_to_ids([a]) for a in rep] )
            except KeyError:
                print("###################")
                return results
            print(inputt)
            caps.append(alter_cap)
            inputs = torch.tensor([inputt.tolist()[0]]+tokenizer(tokenToString(alter_cap))+[inputt.tolist()[-1]])
            print(inputs)
            changed = True
            target = 1
            if (len(inputs.tolist()) == len(inputt.tolist())):
                results.append((inputs.to(device),target,rep))
    return results 

def make_change1(inputt,toker,data,device):

    tokenizer = bert_tokenize(toker)
    t =tokenToString(" ".join(toker.convert_ids_to_tokens(inputt.tolist())))
    changed = False
    target = 0
    results = []
    for caption,alter_cap,rep,_ in data:
        if tokenToString(caption) == t :
            changed = True
            results.append((inputt,target,[]))
    return results

def main(opts):
    print(opts.model,' dec ?',opts.dec,' img_data :',opts.img_db)
    hvd.init()
    device = torch.device("cuda")  # support single GPU only
    
    train_opts = Struct(json.load(open('config/pretrain-alldata-base-8gpu.json')))

   

    img_db = DetectFeatLmdb(opts.img_db,
                            train_opts.conf_th, train_opts.max_bb,
                            train_opts.min_bb, train_opts.num_bb,
                            opts.compressed_db)                   
    txt_db = TxtTokLmdb(opts.txt_db, -1)
   

    if opts.dec != 1:

        da = RPRDataset
        fn = rpr_collate
    else :
        print("decorelated")
        da = RPRDataset_decorrelated
        fn = rpr_collate_dec
    dset = da(txt_db, img_db)
    print(f'RPRDataset len :{len(dset.ids)}')

    batch_size = (train_opts.val_batch_size if opts.batch_size is None
                  else opts.batch_size)
                  
    print(f'batch_size,{batch_size}')

    eval_dataloader = DataLoader(dset, batch_size=batch_size,
                                 num_workers=1,
                                 pin_memory=opts.pin_mem,
                                 collate_fn=fn)
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

    evaluate(model, eval_dataloader, device,opts)
   




@torch.no_grad()
def evaluate(model, eval_loader, device,opts):
    print("start running evaluation...")
    toker = BertTokenizer.from_pretrained("bert-base-cased")
    model.eval()

    n_ex = 0
    st = time()
    results = []
    t0 = time()
    texts =[]
    changed_text = []
    fnames =[]
    tensor_out = []
    tensor_out_pooled=[]
    targets = []
    remplacements = []


    for m in [make_change,make_change1]:
     for  batch in tqdm(eval_loader,total =len(eval_loader.dataset)):
        

        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attention_mask = batch['attn_masks']
        gather_index = batch['gather_index']
        

        
        t = " ".join(toker.convert_ids_to_tokens(batch['input_ids'][0].tolist()))
        results = m(input_ids.squeeze(0),toker,captions,device)
        if results != [] :
            for (input_idss,target,remplacement) in results :
                with torch.no_grad():
                    #output = model(batch, task="getrpr", compute_loss=False)
                    output = model.uniter(input_idss.unsqueeze(0), position_ids,
                                              img_feat, img_pos_feat,
                                              attention_mask, gather_index,
                                              output_all_encoded_layers=False)
                    output_pooled= model.uniter.pooler(output)

                texts.append(t)
                fnames.append(batch['im_fnames'][0])
                tensor_out.append(output.to('cpu'))
                tensor_out_pooled.append(output_pooled.to('cpu'))
                changed_text.append(" ".join(toker.convert_ids_to_tokens(input_idss.tolist())))
                targets.append(target)
                remplacements.append(remplacement)
                del output_pooled
                del output
                torch.cuda.empty_cache()

     print(len(texts),len(fnames))
 
    repre = {"fnames":fnames , "texts":texts ,'representations':tensor_out,'pooled_representation':tensor_out_pooled,
             "changed_text":changed_text ,"targets":targets,"remplacements":remplacements }
    for a,b in repre.items():
         print(a ," ; ",len(b))

    filename = "../representations/uniter_altercaps_"+opts.model+"_coco.pickle" if opts.dec==0 else "../representations/uniter_altercaps_"+opts.model+"_coco_dec.pickle"

    with open(filename, 'wb') as handle:
        pickle.dump(repre, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("saved ",filename)


   


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--txt_db",
                        type=str, required=True,
                        help="The input train corpus.")

    parser.add_argument("--dec",
                        type=int, required=True,
                        help="decorrelated?")
    parser.add_argument("--model",
                        type=str, required=True,
                        help="model.")
    parser.add_argument("--img_db",
                        type=str, required=True,
                        help="The input train images.")

    parser.add_argument('--compressed_db', action='store_true',
                        help='use compressed LMDB')
    parser.add_argument("--batch_size", type=int,default = 1,
                        help="batch size for evaluation")
    parser.add_argument('--n_workers', type=int, default=1,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true',
                        help="pin memory")

    args = parser.parse_args()

    main(args)
