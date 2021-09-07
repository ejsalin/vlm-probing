python get_bert_representations.py --ann ../lxmert/data/mscoco_imgfeat/coco_altercap_lxmert.csv --output ../representations/bert_altercaps_train.pickle
python get_bert_representations.py --ann ../lxmert/data/mscoco_imgfeat/coco_altercap_lxmert_val.csv --output ../representations/bert_altercaps_val.pickle
python get_bert_representations.py --ann ../datasets/coco_lxmert.csv --output ../representations/bert_coco.pickle

python get_bert_representations.py --ann ../datasets/flickr_3000_bshift.csv --output ../representations/bert_bshift.pickle

python get_bert_representations.py --ann ../datasets/flickr_3000_pos.csv --output ../representations/bert_pos.pickle

python get_bert_representations.py --ann ../datasets/flickr_3000_colors.csv --output ../representations/bert_colors.pickle

python get_bert_representations.py --ann ../datasets/flickr_3000_size.csv --output ../representations/bert_size.pickle

python get_bert_representations.py --ann ../datasets/flickr_3000.csv --output ../representations/bert_flickr.pickle
#
#




