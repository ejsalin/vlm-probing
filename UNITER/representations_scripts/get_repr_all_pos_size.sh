horovodrun -np 1 python get_representations.py --img_db ./uniter_data/images_data/flickr30k/ --txt_db ./uniter_data/captions_data/flickr3000_txt_db_size  --model base --data flickr --task size --dec 0
horovodrun -np 1 python get_representations.py --img_db ./uniter_data/images_data/flickr30k/ --txt_db ./uniter_data/captions_data/flickr3000_txt_db_size_dec  --model base --data flickr --task size --dec 1


horovodrun -np 1 python get_representations.py --img_db ./uniter_data/images_data/flickr30k/ --txt_db ./uniter_data/captions_data/flickr3000_txt_db_size  --model vqa --data flickr --task size --dec 0
horovodrun -np 1 python get_representations.py --img_db ./uniter_data/images_data/flickr30k/ --txt_db ./uniter_data/captions_data/flickr3000_txt_db_size_dec  --model vqa --data flickr --task size --dec 1


horovodrun -np 1 python get_representations.py --img_db ./uniter_data/images_data/flickr30k/ --txt_db ./uniter_data/captions_data/flickr3000_txt_db_size  --model nlvr --data flickr --task size --dec 0
horovodrun -np 1 python get_representations.py --img_db ./uniter_data/images_data/flickr30k/ --txt_db ./uniter_data/captions_data/flickr3000_txt_db_size_dec  --model nlvr --data flickr --task size --dec 1

horovodrun -np 1 python get_representations.py --img_db ./uniter_data/images_data/flickr30k/ --txt_db ./uniter_data/captions_data/flickr3000_txt_db_pos  --model base --data flickr --task pos --dec 0
horovodrun -np 1 python get_representations.py --img_db ./uniter_data/images_data/flickr30k/ --txt_db ./uniter_data/captions_data/flickr3000_txt_db_pos_dec  --model base --data flickr --task pos --dec 1


horovodrun -np 1 python get_representations.py --img_db ./uniter_data/images_data/flickr30k/ --txt_db ./uniter_data/captions_data/flickr3000_txt_db_pos  --model vqa --data flickr --task pos --dec 0
horovodrun -np 1 python get_representations.py --img_db ./uniter_data/images_data/flickr30k/ --txt_db ./uniter_data/captions_data/flickr3000_txt_db_pos_dec  --model vqa --data flickr --task pos --dec 1


horovodrun -np 1 python get_representations.py --img_db ./uniter_data/images_data/flickr30k/ --txt_db ./uniter_data/captions_data/flickr3000_txt_db_pos  --model nlvr --data flickr --task pos --dec 0
horovodrun -np 1 python get_representations.py --img_db ./uniter_data/images_data/flickr30k/ --txt_db ./uniter_data/captions_data/flickr3000_txt_db_pos_dec  --model nlvr --data flickr --task pos --dec 1