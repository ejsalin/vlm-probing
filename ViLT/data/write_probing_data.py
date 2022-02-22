import json
import pandas as pd
import pyarrow as pa
import random
import os

from tqdm import tqdm
from glob import glob
from collections import defaultdict


def path2rest(path, iid2captions, iid2split, iid2target, task):
    name = path.split("/")[-1]

    with open(path, "rb") as fp:
        binary = fp.read()
    
    if 'altcap' in task:
        captions = iid2captions[name]
        split = iid2split[name]
        target = iid2target[name]
    else:
        captions = iid2captions[name]
        split = iid2split[name]
        target = iid2target[name]

    return [binary, captions, name, split, target]


def make_arrow(task, dataset_root, data_type):
    if data_type == 'flickr' and task == 'tag':
        captions = pd.read_csv(f"../../probing-vl-models/datasets/flickr_3000.csv", sep='|')
    elif data_type == 'flickr' and task == 'tag_dec':
        captions = pd.read_csv(f"../../probing-vl-models/datasets/flickr_3000_dec.csv", sep='|')
    elif data_type == 'flickr':
        captions = pd.read_csv(f"../../probing-vl-models/datasets/flickr_3000_{task}.csv", sep='|')
    elif data_type == 'coco' and  task=='altcap':
        captions1 = pd.read_csv(f"../../probing-vl-models/datasets/coco_altercap_uniter_train.csv", sep='|')
        captions2 = pd.read_csv(f"../../probing-vl-models/datasets/coco_altercap_uniter_val.csv", sep='|')
        captions = pd.concat([captions1, captions2])
    elif data_type == 'coco' and task=='altcap_dec':
        captions1 = pd.read_csv(f"../../probing-vl-models/datasets/coco_altercap_uniter_train_dec.csv", sep='|')
        captions2 = pd.read_csv(f"../../probing-vl-models/datasets/coco_altercap_uniter_val_dec.csv", sep='|')
        captions = pd.concat([captions1, captions2])

    elif task == 'objcount':
        captions = pd.read_csv(f"../../probing-vl-models/datasets/coco_lxmert.csv", sep='|')
    elif task == 'objcount_dec':
        captions = pd.read_csv(f"../../probing-vl-models/datasets/coco_lxmert_dec.csv", sep='|')


    elif task == 'flowers':
        captions = pd.read_csv(f"../../probing-vl-models/datasets/coco_lxmert_dec.csv", sep='|')

    iid2captions = defaultdict(list)
    iid2split = dict()
    iid2target = dict()
    file_names = []
    for index, cap in tqdm(captions.iterrows()):
        filename = cap["image_name"]
        if data_type == 'coco':
            filename = 'COCO'+ filename[4:]
        if 'altcap' in task:
            i = 0
            while filename+str(i) in file_names:
                i+=1
            filename += str(i)
        iid2split[filename] = "test"
        iid2target[filename] = str(cap["target"])
        iid2captions[filename].append(cap["comment"])
        file_names.append(filename)
   
    if data_type == 'flickr':
        paths = list(glob(f"../../data_all/flickr30k/flickr30k_images/flickr30k_images/*.jpg"))
    if data_type == 'coco':
        paths = list(glob(f"../../data_all/coco/val2014/*.jpg"))
    if data_type == 'flowers':
        paths = list(glob(f"../../data_all/flowers/jpg/*.jpg"))
    caption_paths = []
    for file_name in file_names:
        for path in paths:
            if 'altcap' in task:
                if path.split("/")[-1] == file_name[:-1]:
                    caption_paths.append(path+file_name[-1])
            else:
                if path.split("/")[-1] == file_name:
                    caption_paths.append(path)
    

    if len(paths) == len(caption_paths):
        print("all images have caption annotations")
    else:
        print("not all images have caption annotations")
    print(
        len(paths), len(caption_paths), len(iid2captions), len(iid2target)
    )

    bs = [path2rest(path, iid2captions, iid2split, iid2target, task) for path in tqdm(caption_paths)]

    for split in ["test"]:
        batches = [b for b in bs if b[-2] == split]

        dataframe = pd.DataFrame(
            batches, columns=["image", "caption", "image_id", "split", "target"],
        )
        print("dataframe shape", dataframe.shape)

        table = pa.Table.from_pandas(dataframe)

        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(
            f"{dataset_root}/{data_type}_{task}.arrow", "wb"
        ) as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
