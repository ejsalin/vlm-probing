import argparse
import json
import pickle
import os
from os.path import exists
import re
from cytoolz import curry
from tqdm import tqdm
from pytorch_pretrained_bert import BertTokenizer

from data.data import open_lmdb



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


import csv


def get_id(fname):
    r1 = re.search(r'0[1-9][0-9]+.npz',fname)
    return r1.group(0)[1:-4]
        





def process_flickr(ann, db, tokenizer, missing=None):


    id2len = {}
    txt2img = {}  # not sure if useful
    id2target ={}
    example={}

    i=0
    next(ann) # skip first line
    files = []
    for line in tqdm(ann, desc='processing flickr'):
            id_ = line[0][:-4].zfill(12)+".npz" #get the right name format

            img_fname = id_
            if img_fname == '9279.npz':
                print(img_fname)
                a = input()

            
            if True:

                files.append(id_)
                caption_num = line[1]
                example['sentence'] = line[2]
                input_ids = tokenizer(example['sentence'])
                txt2img[id_] = img_fname+caption_num
                id2len[id_+caption_num] = len(input_ids)
                id2target[id_]= line[-1]
                example['caption_num'] = caption_num
                example['input_ids'] = input_ids
                example['img_fname'] = img_fname
                example['target'] = line[-1]
                db[id_+caption_num] = example
                i+=1

    print("nombre d'example est : ",i)
    return id2len, txt2img,id2target

def main(opts):
    if not exists(opts.output):
        os.makedirs(opts.output)
    else:
        raise ValueError('Found existing DB. Please explicitly remove '
                         'for re-processing')
    meta = vars(opts)
    meta['tokenizer'] = opts.toker
    toker = BertTokenizer.from_pretrained(
        opts.toker, do_lower_case='uncased' in opts.toker)
    tokenizer = bert_tokenize(toker)
    meta['UNK'] = toker.convert_tokens_to_ids(['[UNK]'])[0]
    meta['CLS'] = toker.convert_tokens_to_ids(['[CLS]'])[0]
    meta['SEP'] = toker.convert_tokens_to_ids(['[SEP]'])[0]
    meta['MASK'] = toker.convert_tokens_to_ids(['[MASK]'])[0]
    meta['v_range'] = (toker.convert_tokens_to_ids('!')[0],
                       len(toker.vocab))
    
    with open(f'{opts.output}/meta.json', 'w') as f:
        json.dump(vars(opts), f, indent=4)

    open_db = curry(open_lmdb, opts.output, readonly=False)
    output_field_name = ['id2len', 'txt2img','id2target']
    with open_db() as db:
        print(opts.annotations)
        with open(opts.annotations) as csvfile:
            ann = csv.reader(csvfile, delimiter='|', quotechar='|',)
            jsons = process_flickr(
                ann, db, tokenizer)


    for dump, name in zip(jsons, output_field_name):
        with open(f'{opts.output}/{name}.json', 'w') as f:
            json.dump(dump, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', required=True,
                        help='annotation CSV')

    parser.add_argument('--output', required=True,
                        help='output dir of DB')

    parser.add_argument('--toker', default='bert-base-cased',
                        help='which BERT tokenizer to used')
    args = parser.parse_args()

    main(args)
    
       
