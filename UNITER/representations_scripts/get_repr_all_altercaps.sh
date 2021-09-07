PYTHONNOUSERSITE=1 horovodrun -np 1 python get_representations.py --img_db ../uniter_data/images_data/coco_val2014/ --txt_db ../uniter_data/captions_data/coco_txt_db_altercap_train  --model base --data coco --task train --dec 0
PYTHONNOUSERSITE=1 horovodrun -np 1 python get_representations.py --img_db ../uniter_data/images_data/coco_val2014/ --txt_db ../uniter_data/captions_data/coco_txt_db_altercap_train_dec  --model base --data coco --task train --dec 1


PYTHONNOUSERSITE=1 horovodrun -np 1 python get_representations.py --img_db ../uniter_data/images_data/coco_val2014/ --txt_db ../uniter_data/captions_data/coco_txt_db_altercap_train  --model vqa --data coco --task train --dec 0
PYTHONNOUSERSITE=1 horovodrun -np 1 python get_representations.py --img_db ../uniter_data/images_data/coco_val2014/ --txt_db ../uniter_data/captions_data/coco_txt_db_altercap_train_dec  --model vqa --data coco --task train --dec 1


PYTHONNOUSERSITE=1 horovodrun -np 1 python get_representations.py --img_db ../uniter_data/images_data/coco_val2014/ --txt_db ../uniter_data/captions_data/coco_txt_db_altercap_train  --model nlvr --data coco --task train --dec 0
PYTHONNOUSERSITE=1 horovodrun -np 1 python get_representations.py --img_db ../uniter_data/images_data/coco_val2014/ --txt_db ../uniter_data/captions_data/coco_txt_db_altercap_train_dec  --model nlvr --data coco --task train --dec 1


PYTHONNOUSERSITE=1 horovodrun -np 1 python get_representations.py --img_db ../uniter_data/images_data/coco_val2014/ --txt_db ../uniter_data/captions_data/coco_txt_db_altercap_val  --model base --data coco --task val --dec 0
PYTHONNOUSERSITE=1 horovodrun -np 1 python get_representations.py --img_db ../uniter_data/images_data/coco_val2014/ --txt_db ../uniter_data/captions_data/coco_txt_db_altercap_val_dec  --model base --data coco --task val --dec 1


PYTHONNOUSERSITE=1 horovodrun -np 1 python get_representations.py --img_db ../uniter_data/images_data/coco_val2014/ --txt_db ../uniter_data/captions_data/coco_txt_db_altercap_val  --model vqa --data coco --task val --dec 0
PYTHONNOUSERSITE=1 horovodrun -np 1 python get_representations.py --img_db ../uniter_data/images_data/coco_val2014/ --txt_db ../uniter_data/captions_data/coco_txt_db_altercap_val_dec  --model vqa --data coco --task val --dec 1


PYTHONNOUSERSITE=1 horovodrun -np 1 python get_representations.py --img_db ../uniter_data/images_data/coco_val2014/ --txt_db ../uniter_data/captions_data/coco_txt_db_altercap_val  --model nlvr --data coco --task val --dec 0
PYTHONNOUSERSITE=1 horovodrun -np 1 python get_representations.py --img_db ../uniter_data/images_data/coco_val2014/ --txt_db ../uniter_data/captions_data/coco_txt_db_altercap_val_dec  --model nlvr --data coco --task val --dec 1

