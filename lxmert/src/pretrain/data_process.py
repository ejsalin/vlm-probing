# coding=utf-8
# Copyleft 2019 project LXRT.
from collections import defaultdict
import json
import random
import csv
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from param import args
from pretrain.qa_answer_table import AnswerTable
from utils import load_obj_tsv

TINY_IMG_NUM = 500
FAST_IMG_NUM = 5000

Split2ImgFeatPath = {
    'flickr': 'data/flickrfeatures/train_obj36.tsv',
    'flickr_colors': 'data/flickrfeatures/train_obj36.tsv',
    'flickr_bshift': 'data/flickrfeatures/train_obj36.tsv',
    'flickr_dec': 'data/flickrfeatures/train_obj36.tsv',
    'flickr_colors_dec': 'data/flickrfeatures/train_obj36.tsv',
    'flickr_bshift_dec': 'data/flickrfeatures/train_obj36.tsv',
    'flickr_pos': 'data/flickrfeatures/train_obj36.tsv',
    'flickr_size': 'data/flickrfeatures/train_obj36.tsv',
    'flickr_pos_dec': 'data/flickrfeatures/train_obj36.tsv',
    'flickr_size_dec': 'data/flickrfeatures/train_obj36.tsv',
    'flickr_bw': 'data/flickr_bw_features/train_obj36.tsv',
    'flowers': 'data/flowers/train_obj36.tsv',
    'coco': 'data/mscoco_imgfeat/val2014_obj36.tsv',
    'coco_dec': 'data/mscoco_imgfeat/val2014_obj36.tsv',
    'coco_altercaps_train': 'data/mscoco_imgfeat/val2014_obj36.tsv',
    'coco_altercaps_train_dec': 'data/mscoco_imgfeat/val2014_obj36.tsv',
    'coco_altercaps_val': 'data/mscoco_imgfeat/val2014_obj36.tsv',
    'coco_altercaps_val_dec': 'data/mscoco_imgfeat/val2014_obj36.tsv',

    'mscoco_minival_bw': 'data/coco_val_bw/val2014_obj36.tsv'

}
text_data = {
    'flickr': 'data/flickrfeatures/flickr_3000.csv',
    'flickr_colors': 'data/flickrfeatures/flickr_3000_colors.csv',
    'flickr_bshift': 'data/flickrfeatures/flickr_3000_bshift.csv',
    'flickr_dec': 'data/flickrfeatures/flickr_3000_dec.csv',
    'flickr_colors_dec': 'data/flickrfeatures/flickr_3000_colors_dec.csv',
    'flickr_bshift_dec': 'data/flickrfeatures/flickr_3000_bshift_dec.csv',
    'flickr_pos': 'data/flickrfeatures/flickr_3000_pos.csv',
    'flickr_size': 'data/flickrfeatures/flickr_3000_size.csv',
    'flickr_pos_dec': 'data/flickrfeatures/flickr_3000_pos_dec.csv',
    'flickr_size_dec': 'data/flickrfeatures/flickr_3000_size_dec.csv',
    'flowers': 'data/flowers/flower_result.csv',
    'coco': 'data/mscoco_imgfeat/coco_lxmert.csv',
    'coco_dec': 'data/mscoco_imgfeat/coco_lxmert_dec.csv',

    'coco_altercaps_train': 'data/mscoco_imgfeat/coco_altercap_lxmert.csv',
    'coco_altercaps_train_dec': 'data/mscoco_imgfeat/coco_altercap_lxmert_train_dec.csv',
    'coco_altercaps_val': 'data/mscoco_imgfeat/coco_altercap_lxmert_val.csv',
    'coco_altercaps_val_dec': 'data/mscoco_imgfeat/coco_altercap_lxmert_val_dec.csv',



    

}

class InputExample(object):
    """A single training/test example for the language model."""
    def __init__(self, uid, sent, visual_feats=None,
                 obj_labels=None, attr_labels=None,
                 is_matched=None, label=None,image_id=None):
        self.uid = uid
        self.sent = sent
        self.visual_feats = visual_feats
        self.obj_labels = obj_labels
        self.attr_labels = attr_labels
        self.is_matched = is_matched  # whether the visual and obj matched
        self.label = label #to use as target 
        self.image_id = image_id #added line


class LXMERTDataset:
    def __init__(self, splits: str, qa_sets=None):
        """
        :param splits: The data sources to be loaded
        :param qa_sets: if None, no action
                        o.w., only takes the answers appearing in these dsets
                              and remove all unlabeled data (MSCOCO captions)
        """
        self.name = splits
        self.sources = splits.split(',')

        # Loading datasets to data
        self.data = []
        

        with open(text_data[args.valid]) as csvfile:
            ann = csv.reader(csvfile, delimiter='|', quotechar='|',)

            next(ann) # skip first line
            for line in tqdm(ann, desc='processing flickr'):
                    im_id = line[0][:-4] # delete the .jpg at the end of file name in csv 
                    self.data.append((im_id,line[2],line[3],line[-1]))
        



    def __len__(self):
        return len(self.data)


def make_uid(img_id, dset, sent_idx):
    return "%s_%s_%03d" % (img_id, dset, sent_idx),


"""
Example in obj tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
"""
class LXMERTTorchDataset(Dataset):
    def __init__(self, dataset: LXMERTDataset, topk=-1):
        super().__init__()
        self.raw_dataset = dataset

        self.task_matched = False

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM

        # Load the dataset
        img_data = []
        for source in self.raw_dataset.sources:
            img_data.extend(load_obj_tsv(Split2ImgFeatPath[source]))

        self.imgid2img = {}
        data = []
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum
            data.append(img_datum['img_id'])


        # Filter out the dataset
        used_data = []

        for datum in self.raw_dataset.data:
            if datum[0] in self.imgid2img:
                used_data.append(datum)
            else :
                print(datum[0], " not in the file_imgs")

        # Flatten the dataset (into one sent + one image entries)
        self.data = []
        for datum in used_data:
            sentf = datum[1]
            target = datum[-1]
            new_datum = {
                        'uid': "none",
                        'img_id': datum[0],
                        'sent': sentf,
                        'target': target
                    }
                
            self.data.append(new_datum)

        print("Use %d data in torch dataset" % (len(self.data)))

    def __len__(self):
        return len(self.data)

    def random_feat(self):
        """Get a random obj feat from the dataset."""
        datum = self.data[random.randint(0, len(self.data)-1)]
        img_id = datum['img_id']
        img_info = self.imgid2img[img_id]
        feat = img_info['features'][random.randint(0, 35)]
        return feat

    def __getitem__(self, item: int):
        datum = self.data[item]

        uid = datum['uid']
        img_id = datum['img_id']

        # Get image info
        
        img_info = self.imgid2img[img_id]

        obj_num = img_info['num_boxes']
        feats = img_info['features'].copy()
        boxes = img_info['boxes'].copy()
        obj_labels = img_info['objects_id'].copy()

        obj_confs = img_info['objects_conf'].copy()
        attr_labels = img_info['attrs_id'].copy()
        attr_confs = img_info['attrs_conf'].copy()
        assert obj_num == len(boxes) == len(feats)

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        # If calculating the matched loss, replace the sentence with an sentence
        # corresponding to other image.
        is_matched = 0
        sent = datum['sent']
        target = datum['target']



        # Label, convert answer to id
        #if 'label' in datum:
        #    label = datum['label'].copy()
        #    for ans in list(label.keys()):
        #        label[self.raw_dataset.answer_table.ans2id(ans)] = label.pop(ans)
        #else:
        label = None

        # Create target
        example = InputExample(
            uid, sent, (feats, boxes),
            (obj_labels, obj_confs), (attr_labels, attr_confs),
            is_matched, label = target,image_id=(datum['img_id'])  #img_id added 
        )
        return example


class LXMERTEvaluator:
    def __init__(self, dataset: LXMERTDataset):
        self.raw_dataset = dataset

        # Create QA Eval Data
        self.data = []
        for datum in self.raw_dataset.data:
            sentf = datum['sentf']
            for sents_cat, sents in sentf.items():
                if sents_cat in datum['labelf']:    # A labeled dataset
                    labels = datum['labelf'][sents_cat]
                    for sent_idx, sent in enumerate(sents):
                        new_datum = {
                            'uid': make_uid(datum['img_id'], sents_cat, sent_idx),
                            'img_id': datum['img_id'],
                            'sent': sent,
                            'dset': sents_cat,
                            'label': labels[sent_idx]
                        }
                        self.data.append(new_datum)

        # uid2datum
        self.uid2datum = {}
        for datum in self.data:
            self.uid2datum[datum['uid']] = datum

    def evaluate(self, uid2ans: dict, pprint=False):
        score = 0.
        cnt = 0
        dset2score = defaultdict(lambda: 0.)
        dset2cnt = defaultdict(lambda: 0)
        for uid, ans in uid2ans.items():
            if uid not in self.uid2datum:   # Not a labeled data
                continue
            datum = self.uid2datum[uid]
            label = datum['label']
            dset = datum['dset']
            if ans in label:
                score += label[ans]
                dset2score[dset] += label[ans]
            cnt += 1
            dset2cnt[dset] += 1
        accu = score / cnt
        dset2accu = {}
        for dset in dset2cnt:
            dset2accu[dset] = dset2score[dset] / dset2cnt[dset]

        if pprint:
            accu_str = "Overall Accu %0.4f, " % (accu)
            sorted_keys = sorted(dset2accu.keys())
            for key in sorted_keys:
                accu_str += "%s Accu %0.4f, " % (key, dset2accu[key])
            print(accu_str)

        return accu, dset2accu

    def dump_result(self, uid2ans: dict, path):
        raise NotImplemented


