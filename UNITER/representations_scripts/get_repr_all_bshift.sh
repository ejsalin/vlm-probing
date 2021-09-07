PYTHONNOUSERSITE=1 horovodrun -np 1 python get_representations.py --img_db ../uniter_data/images_data/flickr30k/ --txt_db ../uniter_data/captions_data/flickr3000_txt_db_bshift  --model base --data flickr --task bshift --dec 0
PYTHONNOUSERSITE=1 horovodrun -np 1 python get_representations.py --img_db ../uniter_data/images_data/flickr30k/ --txt_db ../uniter_data/captions_data/flickr3000_txt_db_bshift_dec  --model base --data flickr --task bshift --dec 1


PYTHONNOUSERSITE=1 horovodrun -np 1 python get_representations.py --img_db ../uniter_data/images_data/flickr30k/ --txt_db ../uniter_data/captions_data/flickr3000_txt_db_bshift  --model vqa --data flickr --task bshift --dec 0
PYTHONNOUSERSITE=1 horovodrun -np 1 python get_representations.py --img_db ../uniter_data/images_data/flickr30k/ --txt_db ../uniter_data/captions_data/flickr3000_txt_db_bshift_dec  --model vqa --data flickr --task bshift --dec 1


PYTHONNOUSERSITE=1 horovodrun -np 1 python get_representations.py --img_db ../uniter_data/images_data/flickr30k/ --txt_db ../uniter_data/captions_data/flickr3000_txt_db_bshift  --model nlvr --data flickr --task bshift --dec 0
PYTHONNOUSERSITE=1 horovodrun -np 1 python get_representations.py --img_db ../uniter_data/images_data/flickr30k/ --txt_db ../uniter_data/captions_data/flickr3000_txt_db_bshift_dec  --model nlvr --data flickr --task bshift --dec 1