# Are Vision-Language Transformers Learning Multimodal Representations? A Probing Perspective

This is the repository to reproduce the probing experiments of the paper:
"Are Vision-Language Transformers Learning Multimodal Representations? A Probing Perspective."
This repository is based on implementations by [UNITER](https://github.com/ChenRocks/UNITER) and [LXMERT](https://github.com/airsplay/lxmert).

## Requirements
To reproduce the results for UNITER, use the instructions on the original repository to set up the environment.

To reproduce the results for LXMERT, use the instructions on the original repository to set up the environment.

The baselines are compatible with the LXMERT environment.

## Reproducing Paper Results
To reproduce the results of the experiments on a specific Vision-Language model and probing task:
- Download the data for the probing task
- Compute the image region representations
- Compute the model representations for this dataset
- Execute the probing task
Our precomputed representations can be available on demand

### Data

The datasets and splits are available in the ```data.zip```.

### Models

#### Pre-trained models
- For UNITER, following their instructions :
```sh
cd UNITER
bash scripts/download_pretrained.sh $PATH_TO_STORAGE
```
We use the base models in our experiments.
- For LXMERT, following their instructions, the model is available at http://nlp.cs.unc.edu/data/model_LXRT.pth.

#### Fine-tuned models

For the paper experiments, the models are fine-tuned from pre-trained models on tasks VQA and NLVR2.
UNITER and LXMERT detail their instructions in their repositories.

### Computing the image representations TODO

### Computing the representations from scratch 
To compute the UNITER, LXMERT, BERT, VIT and RESNET representations:

1- For UNITER, execute: 
```sh
cd UNITER
./representations_scripts/get_repr_all_altercaps.sh # for the adversarial caption task  
./representations_scripts/get_repr_all_colors.sh # for the color classification task  
./representations_scripts/get_repr_all_bshift.sh # for the bigramshift task  
./representations_scripts/get_repr_all_pos_size.sh # for the position and  size tasks 
./representations_scripts/get_repr_all_flower.sh # for the flower classification task  
./representations_scripts/get_repr_all.sh # for the pos-tagging task  
./representations_scripts/get_repr_coco.sh # for the coco object counting task  
```
2 - For LXMERT, execute:
```sh
cd lxmert
./run/get_repr_all_altercaps.sh # for the adversarial caption task  
./run/get_repr_all_colors.sh # for the color classification task  
./run/get_repr_all_bshift.sh # for the bigramshift task  
./run/get_repr_all_pos_size.sh # for the position and  size tasks 
./run/get_repr_all_flowers.sh # for the flower classification task  
./run/get_repr_all.sh # for the pos-tagging task  
./run/get_repr_coco.sh # for the coco object counting task  
```

3 - For the baselines models (BERT,VIT and RESNET), execute :
```sh
cd probing_tasks
./get_vit_representation.sh
./get_bert_representation.sh
./get_resnet_representation.sh
```
At the end all representations will be in the representations dir 

You can skip this steps by downloading representations using this script :
TO DO

### Probing

For the probing you have to execute those commandes :
```sh
cd probing_tasks
./altercaps.sh
./bshift.sh
./colors_cls.sh
./flowers.sh
./postag.sh
```



