horovodrun -np 1 python get_representations.py --img_db ./uniter_data/images_data/flowers/ --txt_db ./uniter_data/captions_data/flower_txt_db  --model base --data flower --task org --dec 0

horovodrun -np 1 python get_representations.py --img_db ./uniter_data/images_data/flowers/ --txt_db ./uniter_data/captions_data/flower_txt_db  --model vqa --data flower --task org --dec 0

horovodrun -np 1 python get_representations.py --img_db ./uniter_data/images_data/flowers/ --txt_db ./uniter_data/captions_data/flower_txt_db  --model nlvr --data flower --task org --dec 0