
python probing_objCount.py --vit_data ../representations/vit_coco.pickle --resnet_data ../representations/resnet_coco.pickle --bert_data ../representations/bert_coco.pickle --uniter_data ../representations/uniter_obj_base_coco.pickle --lxmert_data ../representations/lxmert_coco_base.pickle  --exp_name coco_base
python probing_objCount.py --vit_data ../representations/vit_coco.pickle --resnet_data ../representations/resnet_coco.pickle --bert_data ../representations/bert_coco.pickle --uniter_data ../representations/uniter_obj_base_coco_dec.pickle --lxmert_data ../representations/lxmert_coco_dec_base.pickle  --exp_name coco_base_dec
python explore_results.py --path probing_results/coco_base.pickle --output probing_results/coco_base
python explore_results.py --path probing_results/coco_base_dec.pickle --output probing_results/coco_base_dec

python probing_objCount.py --vit_data ../representations/vit_coco.pickle --resnet_data ../representations/resnet_coco.pickle --bert_data ../representations/bert_coco.pickle --uniter_data ../representations/uniter_obj_vqa_coco.pickle --lxmert_data ../representations/lxmert_coco_vqa.pickle  --exp_name coco_vqa
python probing_objCount.py --vit_data ../representations/vit_coco.pickle --resnet_data ../representations/resnet_coco.pickle --bert_data ../representations/bert_coco.pickle --uniter_data ../representations/uniter_obj_vqa_coco_dec.pickle --lxmert_data ../representations/lxmert_coco_dec_vqa.pickle  --exp_name coco_vqa_dec
python explore_results.py --path probing_results/coco_vqa.pickle --output probing_results/coco_vqa
python explore_results.py --path probing_results/coco_vqa_dec.pickle --output probing_results/coco_vqa_dec

python probing_objCount.py --vit_data ../representations/vit_coco.pickle --resnet_data ../representations/resnet_coco.pickle --bert_data ../representations/bert_coco.pickle --uniter_data ../representations/uniter_obj_nlvr_coco.pickle --lxmert_data ../representations/lxmert_coco_nlvr.pickle  --exp_name coco_nlvr
python probing_objCount.py --vit_data ../representations/vit_coco.pickle --resnet_data ../representations/resnet_coco.pickle --bert_data ../representations/bert_coco.pickle --uniter_data ../representations/uniter_obj_nlvr_coco_dec.pickle --lxmert_data ../representations/lxmert_coco_dec_nlvr.pickle  --exp_name coco_nlvr_dec

python explore_results.py --path probing_results/coco_nlvr.pickle --output probing_results/coco_nlvr
python explore_results.py --path probing_results/coco_nlvr_dec.pickle --output probing_results/coco_nlvr_dec
