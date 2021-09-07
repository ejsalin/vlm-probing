# The name of experiment
name=lxmert

# Create dirs and make backup
output=snap/pretrain/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# Pre-training

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:./src \
    python src/pretrain/get_representations.py \
    --valid flowers \
    --dec 0 \
    --model base \
    --data flowers \
    --tqdm --output $output ${@:2}

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:./src \
    python src/pretrain/get_representations.py \
    --valid flowers \
    --dec 0 \
    --model vqa \
    --data flowers \
    --tqdm --output $output ${@:2}

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:./src \
    python src/pretrain/get_representations.py \
    --valid flowers \
    --dec 0 \
    --model nlvr \
    --data flowers \
    --tqdm --output $output ${@:2}
