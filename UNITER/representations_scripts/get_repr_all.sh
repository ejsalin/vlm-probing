horovodrun -np 1 python get_representations.py --img_db ./uniter_data/images_data/flickr30k/ --txt_db ./uniter_data/captions_data/flickr3000_txt_db  --model base --data flickr --task org --dec 0
horovodrun -np 1 python get_representations.py --img_db ./uniter_data/images_data/flickr30k/ --txt_db ./uniter_data/captions_data/flickr3000_txt_db_dec  --model base --data flickr --task org --dec 1


horovodrun -np 1 python get_representations.py --img_db ./uniter_data/images_data/flickr30k/ --txt_db ./uniter_data/captions_data/flickr3000_txt_db  --model vqa --data flickr --task org --dec 0
horovodrun -np 1 python get_representations.py --img_db ./uniter_data/images_data/flickr30k/ --txt_db ./uniter_data/captions_data/flickr3000_txt_db_dec  --model vqa --data flickr --task org --dec 1


horovodrun -np 1 python get_representations.py --img_db ./uniter_data/images_data/flickr30k/ --txt_db ./uniter_data/captions_data/flickr3000_txt_db  --model nlvr --data flickr --task org --dec 0
horovodrun -np 1 python get_representations.py --img_db ./uniter_data/images_data/flickr30k/ --txt_db ./uniter_data/captions_data/flickr3000_txt_db_dec  --model nlvr --data flickr --task org --dec 1