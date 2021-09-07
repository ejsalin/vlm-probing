#python probing_postag.py --bert_data ../representations/bert_flickr.pickle  --lxmert_data ../representations/lxmert_flickr_base.pickle --uniter_data ../representations/uniter_org_base_flickr.pickle  --exp_name postag_flickr_base  --data flickr --dec 0
#python explore_results_acc.py --path probing_results/postag_flickr_base.pickle --output probing_results/postag_flickr_base
#python probing_postag.py --bert_data ../representations/bert_flickr.pickle  --lxmert_data ../representations/lxmert_flickr_dec_base.pickle --uniter_data ../representations/uniter_org_base_flickr_dec.pickle  --exp_name postag_flickr_base_dec  --data flickr --dec 0
#python explore_results_acc.py --path probing_results/postag_flickr_base_dec.pickle --output probing_results/postag_flickr_base_dec
#
#python probing_postag.py --bert_data ../representations/bert_flickr.pickle  --lxmert_data ../representations/lxmert_flickr_vqa.pickle --uniter_data ../representations/uniter_org_vqa_flickr.pickle  --exp_name postag_flickr_vqa  --data flickr --dec 0
#python explore_results_acc.py --path probing_results/postag_flickr_vqa.pickle --output probing_results/postag_flickr_vqa
#python probing_postag.py --bert_data ../representations/bert_flickr.pickle  --lxmert_data ../representations/lxmert_flickr_dec_vqa.pickle --uniter_data ../representations/uniter_org_vqa_flickr_dec.pickle  --exp_name postag_flickr_vqa_dec  --data flickr --dec 0
#python explore_results_acc.py --path probing_results/postag_flickr_vqa_dec.pickle --output probing_results/postag_flickr_vqa_dec
#
#
#python probing_postag.py --bert_data ../representations/bert_flickr.pickle  --lxmert_data ../representations/lxmert_flickr_nlvr.pickle --uniter_data ../representations/uniter_org_nlvr_flickr.pickle  --exp_name postag_flickr_nlvr  --data flickr --dec 0
#python explore_results_acc.py --path probing_results/postag_flickr_nlvr.pickle --output probing_results/postag_flickr_nlvr
python probing_postag.py --bert_data ../representations/bert_flickr.pickle  --lxmert_data ../representations/lxmert_flickr_dec_nlvr.pickle --uniter_data ../representations/uniter_org_nlvr_flickr_dec.pickle  --exp_name postag_flickr_nlvr_dec  --data flickr --dec 0
python explore_results_acc.py --path probing_results/postag_flickr_nlvr_dec.pickle --output probing_results/postag_flickr_nlvr_dec
