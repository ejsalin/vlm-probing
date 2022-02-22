import json
import pandas as pd
import pyarrow as pa
import random
import os
import scipy.io
from tqdm import tqdm
from glob import glob
from collections import defaultdict


def path2rest(path, iid2captions, iid2split, iid2target, task=None):
    name = path.split("/")[-1]

    with open(path, "rb") as fp:
        binary = fp.read()
    
    captions = iid2captions[name]
    split = iid2split[name]
    target = iid2target[name]

    return [binary, captions, name, split, target]


def make_arrow_flowers(dataset_root):
    
    targets = scipy.io.loadmat('../../data_all/flowers/imagelabels.mat')

    iid2captions = defaultdict(list)
    iid2split = dict()
    iid2target = dict()
    file_names = []
    for index, target in tqdm(enumerate(targets['labels'][0])):
        filename = 'image_' + '0'*(5-len(str(index)))+str(index)+'.jpg'
        iid2split[filename] = "test"
        iid2target[filename] = str(target)
        iid2captions[filename].append(" ")
        file_names.append(filename)

    paths = list(glob(f"../../data_all/flowers/jpg/*.jpg"))
    caption_paths = []
    for file_name in file_names:
        for path in paths:
            if path.split("/")[-1] == file_name:
                caption_paths.append(path)
    

    if len(paths) == len(caption_paths):
        print("all images have caption annotations")
    else:
        print("not all images have caption annotations")
    print(
        len(paths), len(caption_paths), len(iid2captions), len(iid2target)
    )

    bs = [path2rest(path, iid2captions, iid2split, iid2target) for path in tqdm(caption_paths)]

    for split in ["test"]:
        batches = [b for b in bs if b[-2] == split]

        dataframe = pd.DataFrame(
            batches, columns=["image", "caption", "image_id", "split", "target"],
        )
        print("dataframe shape", dataframe.shape)

        table = pa.Table.from_pandas(dataframe)

        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(
            f"{dataset_root}/flowers_flowers.arrow", "wb"
        ) as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
