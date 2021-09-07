"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

MLM datasets
"""
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip

from .data import (DetectFeatTxtTokDataset, TxtTokLmdb,
                   pad_tensors, get_gather_index)





class RPRDataset(DetectFeatTxtTokDataset):
    def __init__(self, txt_db, img_db):
        assert isinstance(txt_db, TxtTokLmdb)
        super().__init__(txt_db, img_db)

    def __getitem__(self, i):
        """
        Return:
        - input_ids    : (L, ), i.e., [cls, wd, wd, ..., sep, 0, 0], 0s padded
        - img_feat     : (num_bb, d)
        - img_pos_feat : (num_bb, 7)
        - attn_masks   : (L + num_bb, ), ie., [1, 1, ..., 0, 0, 1, 1]
        - txt_labels   : (L, ), [-1, -1, wid, -1, -1, -1]
        0's padded so that (L + num_bb) % 8 == 0
        """
        example = super().__getitem__(i)

        # text input
        input_ids, txt_labels = self.create_rpr_io(example['input_ids'])

        # img input
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'])
        # image file name 
        im_fname = example['img_fname']
        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

        return input_ids, img_feat, img_pos_feat, attn_masks, txt_labels,im_fname

    def create_rpr_io(self, input_ids):
        input_ids, txt_labels = input_ids,input_ids
                                            
        input_ids = torch.tensor([self.txt_db.cls_]
                                 + input_ids
                                 + [self.txt_db.sep])
        txt_labels = torch.tensor([-1] + txt_labels + [-1])
        return input_ids, txt_labels

class RPRDataset_decorrelated(DetectFeatTxtTokDataset):
    def __init__(self, txt_db, img_db,seed = 0):
        assert isinstance(txt_db, TxtTokLmdb)
        super().__init__(txt_db, img_db)
        self.fnames =[]
        self.seed = seed
        for i in range(len(self.ids)):
            e= super().__getitem__(i)
            self.fnames.append(e['img_fname'])
        

    def __getitem__(self, i):
        
        """
        Return:
        - input_ids    : (L, ), i.e., [cls, wd, wd, ..., sep, 0, 0], 0s padded
        - img_feat     : (num_bb, d)
        - img_pos_feat : (num_bb, 7)
        - attn_masks   : (L + num_bb, ), ie., [1, 1, ..., 0, 0, 1, 1]
        - txt_labels   : (L, ), [-1, -1, wid, -1, -1, -1]
        0's padded so that (L + num_bb) % 8 == 0
        """
        example = super().__getitem__(i)

        # text input
        input_ids, txt_labels = self.create_rpr_io(example['input_ids'])

        fname = random.sample(self.fnames,1)[0]
        while fname ==example['img_fname'] :
            fname = random.sample(self.fnames,1)[0]
        # img input
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            fname)
        # image file name 
        im_fname = fname
        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

        return input_ids, img_feat, img_pos_feat, attn_masks, txt_labels,im_fname,example['img_fname']

    def create_rpr_io(self, input_ids):
        input_ids, txt_labels = input_ids,input_ids
                                            
        input_ids = torch.tensor([self.txt_db.cls_]
                                 + input_ids
                                 + [self.txt_db.sep])
        txt_labels = torch.tensor([-1] + txt_labels + [-1])
        return input_ids, txt_labels




def rpr_collate_dec(inputs):
    """
    Return:
    :input_ids    (n, max_L) padded with 0
    :position_ids (n, max_L) padded with 0
    :txt_lens     list of [txt_len]
    :img_feat     (n, max_num_bb, feat_dim)
    :img_pos_feat (n, max_num_bb, 7)
    :num_bbs      list of [num_bb]
    :attn_masks   (n, max_{L + num_bb}) padded with 0
    :txt_labels   (n, max_L) padded with -1
    """
    
    (input_ids, img_feats, img_pos_feats, attn_masks, txt_labels,im_fnames,txt_fnames
     ) = map(list, unzip(inputs))

    # text batches
    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    txt_labels = pad_sequence(txt_labels, batch_first=True, padding_value=-1)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    # image batches
    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'txt_labels': txt_labels,
             'im_fnames':im_fnames,
             "txt_fnames":txt_fnames}
    return batch
