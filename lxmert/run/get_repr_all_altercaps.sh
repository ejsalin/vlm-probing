# The name of experiment
name=lxmert

# Create dirs and make backup
output=snap/pretrain/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash


CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:./src \
    python src/pretrain/get_representations.py \
    --valid coco_altercaps_val \
    --model base \
    --tqdm --output $output ${@:2}

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:./src \
    python src/pretrain/get_representations.py \
    --valid coco_altercaps_val \
    --model vqa \
    --tqdm --output $output ${@:2}

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:./src \
    python src/pretrain/get_representations.py \
    --valid coco_altercaps_val \
    --model nlvr \
    --tqdm --output $output ${@:2}



CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:./src \
    python src/pretrain/get_representations.py \
    --valid coco_altercaps_val_dec \
    --model base \
    --tqdm --output $output ${@:2}

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:./src \
    python src/pretrain/get_representations.py \
    --valid coco_altercaps_val_dec \
    --model vqa \
    --tqdm --output $output ${@:2}

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:./src \
    python src/pretrain/get_representations.py \
    --valid coco_altercaps_val_dec \
    --model nlvr \
    --tqdm --output $output ${@:2}


CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:./src \
    python src/pretrain/get_representations.py \
    --valid coco_altercaps_train \
    --model base \
    --tqdm --output $output ${@:2}

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:./src \
    python src/pretrain/get_representations.py \
    --valid coco_altercaps_train \
    --model vqa \
    --tqdm --output $output ${@:2}

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:./src \
    python src/pretrain/get_representations.py \
    --valid coco_altercaps_train \
    --model nlvr \
    --tqdm --output $output ${@:2}



CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:./src \
    python src/pretrain/get_representations.py \
    --valid coco_altercaps_train_dec \
    --model base \
    --tqdm --output $output ${@:2}

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:./src \
    python src/pretrain/get_representations.py \
    --valid coco_altercaps_train_dec \
    --model vqa \
    --tqdm --output $output ${@:2}

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:./src \
    python src/pretrain/get_representations.py \
    --valid coco_altercaps_train_dec \
    --model nlvr \
    --tqdm --output $output ${@:2}