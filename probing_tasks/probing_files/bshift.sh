
python probing_bshift.py  --bert_data ../representations/bert_bshift.pickle --uniter_data ../representations/uniter_bshift_base_flickr.pickle --lxmert_data ../representations/../representations/lxmert_flickr_bshift_base.pickle  --exp_name bshift_flickr_base
python explore_results_acc.py --path probing_results/bshift_flickr_base.pickle --output probing_results/bshift_flickr_base 
python probing_bshift.py  --bert_data ../representations/bert_bshift.pickle --uniter_data ../representations/uniter_bshift_nlvr_flickr.pickle --lxmert_data ../representations/../representations/lxmert_flickr_bshift_nlvr.pickle  --exp_name bshift_flickr_nlvr
python explore_results_acc.py --path probing_results/bshift_flickr_nlvr.pickle --output probing_results/bshift_flickr_nlvr 
python probing_bshift.py  --bert_data ../representations/bert_bshift.pickle --uniter_data ../representations/uniter_bshift_vqa_flickr.pickle --lxmert_data ../representations/../representations/lxmert_flickr_bshift_vqa.pickle  --exp_name bshift_flickr_vqa
python explore_results_acc.py --path probing_results/bshift_flickr_vqa.pickle --output probing_results/bshift_flickr_vqa 

python probing_bshift.py  --bert_data ../representations/bert_bshift.pickle --uniter_data ../representations/uniter_bshift_base_flickr_dec.pickle --lxmert_data ../representations/../representations/lxmert_flickr_bshift_dec_base.pickle  --exp_name bshift_flickr_base_dec
python explore_results_acc.py --path probing_results/bshift_flickr_base_dec.pickle --output probing_results/bshift_flickr_base_dec 
python probing_bshift.py  --bert_data ../representations/bert_bshift.pickle --uniter_data ../representations/uniter_bshift_nlvr_flickr_dec.pickle --lxmert_data ../representations/../representations/lxmert_flickr_bshift_dec_nlvr.pickle  --exp_name bshift_flickr_nlvr_dec
python explore_results_acc.py --path probing_results/bshift_flickr_nlvr_dec.pickle --output probing_results/bshift_flickr_nlvr_dec 
python probing_bshift.py  --bert_data ../representations/bert_bshift.pickle --uniter_data ../representations/uniter_bshift_vqa_flickr_dec.pickle --lxmert_data ../representations/../representations/lxmert_flickr_bshift_dec_vqa.pickle  --exp_name bshift_flickr_vqa_dec
python explore_results_acc.py --path probing_results/bshift_flickr_vqa_dec.pickle --output probing_results/bshift_flickr_vqa_dec 
