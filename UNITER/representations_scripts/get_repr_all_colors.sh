PYTHONNOUSERSITE=1 horovodrun -np 1 python get_representations.py --img_db ../uniter_data/images_data/flickr30k/ --txt_db ../uniter_data/captions_data/flickr3000_txt_db_colors  --model base --data flickr --task color --dec 0
PYTHONNOUSERSITE=1 horovodrun -np 1 python get_representations.py --img_db ../uniter_data/images_data/flickr30k/ --txt_db ../uniter_data/captions_data/flickr3000_txt_db_colors_dec  --model base --data flickr --task color --dec 1


PYTHONNOUSERSITE=1 horovodrun -np 1 python get_representations.py --img_db ../uniter_data/images_data/flickr30k/ --txt_db ../uniter_data/captions_data/flickr3000_txt_db_colors  --model vqa --data flickr --task color --dec 0
PYTHONNOUSERSITE=1 horovodrun -np 1 python get_representations.py --img_db ../uniter_data/images_data/flickr30k/ --txt_db ../uniter_data/captions_data/flickr3000_txt_db_colors_dec  --model vqa --data flickr --task color --dec 1


PYTHONNOUSERSITE=1 horovodrun -np 1 python get_representations.py --img_db ../uniter_data/images_data/flickr30k/ --txt_db ../uniter_data/captions_data/flickr3000_txt_db_colors  --model nlvr --data flickr --task color --dec 0
PYTHONNOUSERSITE=1 horovodrun -np 1 python get_representations.py --img_db ../uniter_data/images_data/flickr30k/ --txt_db ../uniter_data/captions_data/flickr3000_txt_db_colors_dec  --model nlvr --data flickr --task color --dec 1