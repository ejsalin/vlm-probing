
python probing_pos_size.py --vit_data ../representations/vit_size.pickle --resnet_data ../representations/resnet_size.pickle  --bert_data ../representations/bert_size.pickle --uniter_data ../representations/uniter_size_base_flickr.pickle --lxmert_data ../representations/../representations/lxmert_flickr_size_base.pickle  --exp_name size_flickr_base --task size
python explore_results_acc.py --path probing_results/size_flickr_base.pickle --output probing_results/size_flickr_base 
python probing_pos_size.py --vit_data ../representations/vit_size.pickle --resnet_data ../representations/resnet_size.pickle  --bert_data ../representations/bert_size.pickle --uniter_data ../representations/uniter_size_vqa_flickr.pickle --lxmert_data ../representations/../representations/lxmert_flickr_size_vqa.pickle  --exp_name size_flickr_vqa --task size
python explore_results_acc.py --path probing_results/size_flickr_vqa.pickle --output probing_results/size_flickr_vqa 

python probing_pos_size.py --vit_data ../representations/vit_size.pickle --resnet_data ../representations/resnet_size.pickle  --bert_data ../representations/bert_size.pickle --uniter_data ../representations/uniter_size_nlvr_flickr.pickle --lxmert_data ../representations/../representations/lxmert_flickr_size_nlvr.pickle  --exp_name size_flickr_nlvr --task size
python explore_results_acc.py --path probing_results/size_flickr_nlvr.pickle --output probing_results/size_flickr_nlvr 

python probing_pos_size.py --vit_data ../representations/vit_size.pickle --resnet_data ../representations/resnet_size.pickle  --bert_data ../representations/bert_size.pickle --uniter_data ../representations/uniter_size_base_flickr_dec.pickle --lxmert_data ../representations/../representations/lxmert_flickr_size_dec_base.pickle  --exp_name size_flickr_base_dec --task size
python explore_results_acc.py --path probing_results/size_flickr_base_dec.pickle --output probing_results/size_flickr_base_dec 
python probing_pos_size.py --vit_data ../representations/vit_size.pickle --resnet_data ../representations/resnet_size.pickle  --bert_data ../representations/bert_size.pickle --uniter_data ../representations/uniter_size_vqa_flickr_dec.pickle --lxmert_data ../representations/../representations/lxmert_flickr_size_dec_vqa.pickle  --exp_name size_flickr_vqa_dec --task size
python explore_results_acc.py --path probing_results/size_flickr_vqa_dec.pickle --output probing_results/size_flickr_vqa_dec 

python probing_pos_size.py --vit_data ../representations/vit_size.pickle --resnet_data ../representations/resnet_size.pickle  --bert_data ../representations/bert_size.pickle --uniter_data ../representations/uniter_size_nlvr_flickr_dec.pickle --lxmert_data ../representations/../representations/lxmert_flickr_size_dec_nlvr.pickle  --exp_name size_flickr_nlvr_dec --task size
python explore_results_acc.py --path probing_results/size_flickr_nlvr_dec.pickle --output probing_results/size_flickr_nlvr_dec 