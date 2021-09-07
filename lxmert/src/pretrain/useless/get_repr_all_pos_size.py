# coding=utf-8
# Copyleft 2019 project LXRT.
# 
import collections
import os
import random
import pickle 
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
import json
import re
from param import args
from tasks.nlvr2_model import NLVR2Model
from tasks.vqa_model import VQAModel
from difflib import SequenceMatcher


if args.data == "coco":
    from pretrain.get_repr_data import InputExample, LXMERTDataset,LXMERTTorchDataset_deccorellated, LXMERTTorchDataset, LXMERTEvaluator
else :
    from pretrain.lxmert_flicker_data import InputExample, LXMERTDataset,LXMERTTorchDataset_deccorellated, LXMERTTorchDataset, LXMERTEvaluator

from lxrt.entry import set_visual_config
from lxrt.tokenization import BertTokenizer
from lxrt.modeling import LXRTPretraining

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

if args.task=="size":
        words ={"large":"small","small":"tall","little":"big","big":"little","tall":"small","long":"short","short":"long"}
if args.task=="position":
        words ={"top":"bottom","bottom":"top","front":"back","back":"front","under":"above","above":"under","beside":"beyond","beyond":"beside"}

def get_nbobj(fname,id2nb,args):
    if args.data == "coco":
        return id2nb[int(get_id(fname+".npz"))]
    else :
        return id2nb[fname+".xml"]
DataTuple = collections.namedtuple("DataTuple", 'dataset torchdset loader evaluator')


def get_tuple(splits: str, bs: int, shuffle=False, drop_last=False, topk=-1) -> DataTuple:
    qa_sets = args.qa_sets
    if qa_sets is not None:
        qa_sets = set(qa_set.lower().strip() for qa_set in qa_sets.split(","))

    # Build dataset, data loader, and evaluator.
    dset = LXMERTDataset(splits, qa_sets=qa_sets)
    if args.dec == 1:
        tset = LXMERTTorchDataset_deccorellated(dset, topk)
    if args.dec == 0:
        tset = LXMERTTorchDataset(dset, topk)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        collate_fn=lambda x: x,
        drop_last=drop_last, pin_memory=True
    )

    evaluator = None
    print()

    return DataTuple(dataset=dset, torchdset=tset, loader=data_loader, evaluator=evaluator)


#train_tuple = get_tuple(args.train, args.batch_size, shuffle=True, drop_last=True)
valid_batch_size = 1
valid_tuple = get_tuple(args.valid, valid_batch_size, shuffle=True, drop_last=False, topk=5000)
print(valid_tuple)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids, input_mask, segment_ids, lm_label_ids,
                 visual_feats, obj_labels,
                 is_matched, ans,image_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids
        self.visual_feats = visual_feats
        self.obj_labels = obj_labels
        self.is_matched = is_matched
        self.ans = ans
        self.image_id= image_id

    def display(self):
        
        print(self.input_ids )
        print(self.input_mask) 
        print(self.segment_ids) 
        print(self.lm_label_ids) 
        print(self.visual_feats) 
        print(self.obj_labels) 
        print(self.is_matched) 
        print(self.ans )
        print(self.image_id)
             



def convert_example_to_features(example: InputExample, max_seq_length, tokenizer)->InputFeatures:
    """
    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
    IDs, LM labels, input_mask, CLS and SEP tokens etc.
    :param example: InputExample, containing sentence input as strings and is_next label
    :param max_seq_length: int, maximum length of sequence.
    :param tokenizer: Tokenizer
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """

    tokens = tokenizer.tokenize(example.sent.strip())

    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[:(max_seq_length - 2)]

    # Ge random words
    #masked_tokens, masked_label = random_word(tokens, tokenizer)
    ### del the token masking part 
    masked_tokens, masked_label = tokens, tokens
    # concatenate lm labels and account for CLS, SEP, SEP
    masked_tokens = ['[CLS]'] + masked_tokens + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(masked_tokens)

    # Mask & Segment Word
    lm_label_ids = ([-1] + masked_label + [-1])
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        lm_label_ids.append(-1)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(lm_label_ids) == max_seq_length

    feat, boxes = example.visual_feats
    obj_labels, obj_confs = example.obj_labels
    attr_labels, attr_confs = example.attr_labels

    # Mask Image Features:
    #masked_feat, feat_mask = random_feat(feat)

    ### delete feature masking part 
    masked_feat, feat_mask = feat , feat


    # QA answer label
    if example.label is None or len(example.label) == 0 or example.is_matched != 1:
        # 1. No label 2. Label is pruned 3. unmatched visual + language pair
        ans = -1
    else:
        keys, values = zip(*example.label.items())
        if len(keys) == 1:
            ans = keys[0]
        else:
            value_sum = sum(values)
            prob = [value / value_sum for value in values]
            choice = np.random.multinomial(1, prob).argmax()
            ans = keys[choice]

    features = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        lm_label_ids=lm_label_ids,
        visual_feats=(masked_feat, boxes),
        obj_labels={
            'obj': (obj_labels, obj_confs),
            'attr': (attr_labels, attr_confs),
            'feat': (feat, feat_mask),
        },
        is_matched=example.is_matched,
        ans=ans,
        image_id=example.image_id
    )
    return features


LOSSES_NAME = ('Mask_LM', 'Matched', 'Obj', 'Attr', 'Feat', 'QA')


def shift(tensor,words,inv,name,r):



    #print(r)
    if r > 0.5:
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
            #print(tensor_list)
        #print(tensor_list)
        return torch.tensor(tensor_list),1,both
    else:

        return tensor ,0,[]

        



class LXMERT:
    
    def __init__(self, max_seq_length):
        super().__init__()
        if args.model == "base":
            self.max_seq_length = max_seq_length

            self.tokenizer = BertTokenizer.from_pretrained(
                "bert-base-uncased",
                do_lower_case=True
            )

            # Build model
            set_visual_config(args)
            self.model = LXRTPretraining.from_pretrained(
                "bert-base-uncased",
                task_mask_lm=False,
                task_obj_predict=False,
                task_matched=False,
                task_qa=False,
                visual_losses=False,
                #num_answers=train_tuple.dataset.answer_table.num_answers
            )
            # Load lxmert would not load the answer head.
            self.load_lxmert("./snap/pretrained/model_LXRT.pth")
            #self.load("./snap/pretrained/model_LXRT.pth")

            # GPU Options
            self.model = self.model.cuda()
            if args.multiGPU:
                self.model = nn.DataParallel(self.model)
            self.inf_model = self.model.bert
            print("base_model loaded")

        if args.model == "nlvr":
            self.max_seq_length = max_seq_length
                    # Build LXRT encoder
            self.model = NLVR2Model()

            self.tokenizer = BertTokenizer.from_pretrained(
                "bert-base-uncased",
                do_lower_case=True
            )

            #GPU Options
            self.model = self.model.cuda()
            if args.multiGPU:
                self.model = nn.DataParallel(self.model)
            self.load_vqa("./snap/pretrained/nlvr2_lxr955/BEST")
            self.inf_model = self.model.lxrt_encoder.model.bert
            print("nlvr_model loaded")

        if args.model =="vqa":
            self.max_seq_length = max_seq_length
                    # Build LXRT encoder
            self.model = VQAModel(3129)

            self.tokenizer = BertTokenizer.from_pretrained(
                "bert-base-uncased",
                do_lower_case=True
            )

            #GPU Options
            self.model = self.model.cuda()
            if args.multiGPU:
                self.model = nn.DataParallel(self.model)
            self.load_vqa("./snap/pretrained/vqa_lxr955/BEST")
            print("vqa_model loaded")
            self.inf_model = self.model.lxrt_encoder.model.bert
        self.color_idx  = self.tokenizer.convert_tokens_to_ids(words)
        self.inv = {} 
        self.words = list(words.keys())

        for a,b in words.items():
            self.inv[self.tokenizer.convert_tokens_to_ids([a])[0]] = self.tokenizer.convert_tokens_to_ids([b])[0] 
        self.words_ = [self.tokenizer.convert_tokens_to_ids([a])[0] for a in list(words.keys())]


    def forward(self, examples):

        train_features = [convert_example_to_features(example, self.max_seq_length, self.tokenizer)
                          for example in examples]

        # language Inputs
        input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long).cuda()
        print(input_ids)
        input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long).cuda()
        print(input_mask)
        segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long).cuda()

        # Visual Inputs
        feats = torch.from_numpy(np.stack([f.visual_feats[0] for f in train_features])).cuda()
        pos = torch.from_numpy(np.stack([f.visual_feats[1] for f in train_features])).cuda()


        """
        forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                visual_feats=None, pos=None, obj_labels=None, matched_label=None, ans=None):
        """
        loss, losses, ans_logit = self.model(
            input_ids, segment_ids, input_mask, lm_labels,
            feats, pos, obj_labels, matched_labels, ans
        )
        return loss, losses.detach().cpu(), ans_logit

    def get_representation(self,eval_tuple: DataTuple,opts):
        nb= 0
        i = 0
        filename = "../data_used/"+opts.task+"_"+opts.data+".pickle" 
        with open(filename, 'rb') as handle:
            data_ = pickle.load(handle)
            
        self.model.eval()
        eval_ld = eval_tuple.loader
        polled ,visual ,lang, fnames ,texts,targets,masked_sentences,bert_outputs= [],[],[],[],[],[],[],[]
        print(f"len = {len(eval_ld)}")

        for examples in tqdm(eval_ld, total=len(eval_ld)):
                train_features = [convert_example_to_features(example, self.max_seq_length, self.tokenizer)
                          for example in examples]

        # language Inputs
                input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long).cuda()

                text = " ".join(self.tokenizer.convert_ids_to_tokens([f.input_ids for f in train_features][0]))
                if args.dec == 1:
                    fname=[f.image_id[0].lower() for f in train_features][0]
                else:
                    fname=[f.image_id.lower() for f in train_features][0]

                
                if len(set.intersection(set(text.split(" ")),set(self.words))):

                    if tokenToString(text) in data_.keys():
                        nb+=1
                        a = data_[tokenToString(text)]
                        input_ids_,target,shifted_id = shift(input_ids.squeeze(0),self.words_,self.inv,fname,a[1])
                    else :
                        r = np.random.rand(1)[0]
                        input_ids_,target,shifted_id = shift(input_ids.squeeze(0),self.words_,self.inv,fname,r)


                    
                    #print(r)

                    masked = " ".join(self.tokenizer.convert_ids_to_tokens(input_ids_.tolist()))
                    #print(text)
                    #print(masked)
                    #print(target)


                    input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long).cuda()

                    
                    segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long).cuda()


                    feats = torch.from_numpy(np.stack([f.visual_feats[0] for f in train_features])).cuda()

                    pos = torch.from_numpy(np.stack([f.visual_feats[1] for f in train_features])).cuda()


                    with torch.no_grad():
                        (lang_output, visn_output), pooled_output,bert_output = self.inf_model(
                        input_ids_.unsqueeze(0).cuda(), segment_ids, input_mask,
                        visual_feats=(feats, pos),
                        )
                    texts.append(text)
                    polled.append(pooled_output)
                    visual.append(visn_output.to("cpu"))
                    lang.append(lang_output.to("cpu"))
                    fnames.append(fname)
                    masked_sentences.append(masked)
                    targets.append(target)
                    del lang_output
                    del visn_output
                    del pooled_output
                    torch.cuda.empty_cache()

        repre = {"fnames":fnames , "texts":texts ,'visual':visual,'pooled_representation':polled,"lang":lang,
                "masked_sentences":masked_sentences,"targets":targets}
        print("texamples in common :",nb)
        print("lxmert_traget_train",np.mean(np.array(repre['targets'])))
        print('total examples :',len(fnames))
        filename = "../representations/lxmert_bincls_"+args.task+"_"+args.model+"_"+args.data+".pickle" if args.dec==0 else "../representations/lxmert_bincls_"+args.task+"_"+args.model+"_"+args.data+"_dec.pickle"

        with open(filename, 'wb') as handle:
            pickle.dump(repre, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("saving ",filename)

    def load(self, path):
        print("Load BERT extractor from %s" % path)
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)
    def load_vqa(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)
    def load_lxmert(self, path):
        print("Load LXMERT model from %s" % path)
        state_dict = torch.load(path)

        # Do not load any answer head
        for key in list(state_dict.keys()):
            if 'answer' in key:
                state_dict.pop(key)

        # Change Multi GPU to single GPU
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[len("module."):]] = value
        state_dict = new_state_dict

        load_keys = set(state_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        print()
        print("Keys in loaded but not in model:")
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print("Keys in model but not in loaded:")
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()

        self.model.load_state_dict(state_dict, strict=False)

def get_id(fname):
    r1 = re.search(r'0[1-9][0-9]+.npz',fname)
    return r1.group(0)[1:-4]

if __name__ == "__main__":


    
    lxmert = LXMERT(max_seq_length=20)


    lxmert.get_representation(valid_tuple,args)
