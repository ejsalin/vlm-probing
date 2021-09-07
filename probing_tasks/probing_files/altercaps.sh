
python probing_altercaps.py   --bert_data_train ../representations/bert_altercaps_train.pickle --bert_data_val ../representations/bert_altercaps_val.pickle --uniter_data_train ../representations/uniter_train_base_coco.pickle --uniter_data_val ../representations/uniter_val_base_coco.pickle --lxmert_data_train ../representations/lxmert_coco_altercaps_train_base.pickle --lxmert_data_val ../representations/lxmert_coco_altercaps_val_base.pickle  --exp_name altercaps_base 
python probing_altercaps.py   --bert_data_train ../representations/bert_altercaps_train.pickle --bert_data_val ../representations/bert_altercaps_val.pickle --uniter_data_train ../representations/uniter_train_vqa_coco.pickle --uniter_data_val ../representations/uniter_val_vqa_coco.pickle --lxmert_data_train ../representations/lxmert_coco_altercaps_train_vqa.pickle --lxmert_data_val ../representations/lxmert_coco_altercaps_val_vqa.pickle  --exp_name altercaps_vqa 
python probing_altercaps.py   --bert_data_train ../representations/bert_altercaps_train.pickle --bert_data_val ../representations/bert_altercaps_val.pickle --uniter_data_train ../representations/uniter_train_nlvr_coco.pickle --uniter_data_val ../representations/uniter_val_nlvr_coco.pickle --lxmert_data_train ../representations/lxmert_coco_altercaps_train_nlvr.pickle --lxmert_data_val ../representations/lxmert_coco_altercaps_val_nlvr.pickle  --exp_name altercaps_nlvr 

python explore_results_acc.py --path probing_results/altercaps_base.pickle --output probing_results/altercaps_base
python explore_results_acc.py --path probing_results/altercaps_vqa.pickle --output probing_results/altercaps_vqa
python explore_results_acc.py --path probing_results/altercaps_nlvr.pickle --output probing_results/altercaps_nlvr


python probing_altercaps.py   --bert_data_train ../representations/bert_altercaps_train.pickle --bert_data_val ../representations/bert_altercaps_val.pickle --uniter_data_train ../representations/uniter_train_base_coco_dec.pickle --uniter_data_val ../representations/uniter_val_base_coco_dec.pickle --lxmert_data_train ../representations/lxmert_coco_altercaps_train_dec_base.pickle --lxmert_data_val ../representations/lxmert_coco_altercaps_val_dec_base.pickle  --exp_name altercaps_base_dec 
python probing_altercaps.py   --bert_data_train ../representations/bert_altercaps_train.pickle --bert_data_val ../representations/bert_altercaps_val.pickle --uniter_data_train ../representations/uniter_train_vqa_coco_dec.pickle --uniter_data_val ../representations/uniter_val_vqa_coco_dec.pickle --lxmert_data_train ../representations/lxmert_coco_altercaps_train_dec_vqa.pickle --lxmert_data_val ../representations/lxmert_coco_altercaps_val_dec_vqa.pickle  --exp_name altercaps_vqa_dec 
python probing_altercaps.py   --bert_data_train ../representations/bert_altercaps_train.pickle --bert_data_val ../representations/bert_altercaps_val.pickle --uniter_data_train ../representations/uniter_train_nlvr_coco_dec.pickle --uniter_data_val ../representations/uniter_val_nlvr_coco_dec.pickle --lxmert_data_train ../representations/lxmert_coco_altercaps_train_dec_nlvr.pickle --lxmert_data_val ../representations/lxmert_coco_altercaps_val_dec_nlvr.pickle  --exp_name altercaps_nlvr_dec 

python explore_results_acc.py --path probing_results/altercaps_base_dec.pickle --output probing_results/altercaps_base_dec
python explore_results_acc.py --path probing_results/altercaps_vqa_dec.pickle --output probing_results/altercaps_vqa_dec
python explore_results_acc.py --path probing_results/altercaps_nlvr_dec.pickle --output probing_results/altercaps_nlvr_dec

