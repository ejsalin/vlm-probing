horovodrun -np 1 python get_representations.py --img_db ./uniter_data/images_data/coco_val2014/ --txt_db ./uniter_data/captions_data/coco_txt_db  --model base --data coco --task obj --dec 0
horovodrun -np 1 python get_representations.py --img_db ./uniter_data/images_data/coco_val2014/ --txt_db ./uniter_data/captions_data/coco_txt_db_dec  --model base --data coco --task obj --dec 1


horovodrun -np 1 python get_representations.py --img_db ./uniter_data/images_data/coco_val2014/ --txt_db ./uniter_data/captions_data/coco_txt_db  --model vqa --data coco --task obj --dec 0
horovodrun -np 1 python get_representations.py --img_db ./uniter_data/images_data/coco_val2014/ --txt_db ./uniter_data/captions_data/coco_txt_db_dec  --model vqa --data coco --task obj --dec 1


horovodrun -np 1 python get_representations.py --img_db ./uniter_data/images_data/coco_val2014/ --txt_db ./uniter_data/captions_data/coco_txt_db  --model nlvr --data coco --task obj --dec 0
horovodrun -np 1 python get_representations.py --img_db ./uniter_data/images_data/coco_val2014/ --txt_db ./uniter_data/captions_data/coco_txt_db_dec  --model nlvr --data coco --task obj --dec 1

