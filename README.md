# Are Vision-Language Transformers Learning Multimodal Representations? A Probing Perspective

This is the implementations of the experiences in the paper [paper]

## Projet Setup
To reproduces the results of the experiences we can proceed in two ways :
- Computing the representations from scratch 
- Using our precomputed representations 

## Data and models Download
## TODO 
### Computing the representations from scratch 
To compute the UNITER,LXMERT,BERT,VIT and RESNET representations :
1- Use the instruction on [UNITER](https://github.com/ChenRocks/UNITER) original repository to set up the environment , then execute 
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
2 - For Lxmert Representations , Use the instruction on [LXMERT](https://github.com/airsplay/lxmert) original repository to set up the environment , then execute
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

3 - For the baselines models (BERT,VIT and RESNET) use the same env with lxmert and execute :
```sh
cd probing_tasks
./get_vit_representation.sh
./get_bert_representation.sh
./get_resnet_representation.sh
```
At the end all representations will be in the representations dir 

You can skip this steps by downloading representations using this script :
TO DO

## Probing

For the probing you have to execute those commandes :
```sh
cd probing_tasks
./altercaps.sh
./bshift.sh
./colors_cls.sh
./flowers.sh
./postag.sh
```



