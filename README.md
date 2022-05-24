# Are Vision-Language Transformers Learning Multimodal Representations? A Probing Perspective

This is the repository to reproduce the probing experiments of the paper:
"Are Vision-Language Transformers Learning Multimodal Representations? A Probing Perspective."
This repository is based on implementations by [UNITER](https://github.com/ChenRocks/UNITER), [LXMERT](https://github.com/airsplay/lxmert) and [ViLT](https://github.com/dandelin/ViLT).

Data for all experiments is made available in the datasets folder.


## Requirements
To reproduce the results for UNITER, use the instructions on the original repository to set up the environment.

To reproduce the results for LXMERT, use the instructions on the original repository to set up the environment.

To reproduce the results for ViLT, use the instructions on the original repository to set up the environment.

The baselines are compatible with the LXMERT environment.

## Reproducing Paper Results
To reproduce the results of the experiments on a specific Vision-Language model and probing task:
- Download the data for the relevant probing task

- Compute the image region representations following the model instructions

- Compute the joint image/text representations using the model for this dataset (it amounts to the last layer of the transformer)

- Train the probing task on the representations and labels

### Data

We make available new datasets for multimodal probing experiments.

The datasets are available in the ```datasets``` folder. Read ```dataset/README.md``` for more details on the datasets.

The splits are provided in ```probing-vl-models/probing_tasks/split```

### Models

The UNITER models should be saved in ```UNITER/pretrained/$MODEL```

The LXMERT models should be saved in ```lxmert/snap/pretrained/$MODEL```

The ViLT models should be saved in  ```UNITER/checkpoints/$MODEL```

#### Pre-trained models
- For UNITER, following their instructions :
```sh
cd UNITER
bash scripts/download_pretrained.sh $PATH_TO_STORAGE
```
We use the base models in our experiments.
- For LXMERT, following their instructions, the model is available at http://nlp.cs.unc.edu/data/model_LXRT.pth.

We use the pre-trained ViLT model made available by authors in our experiments [here](https://github.com/dandelin/ViLT/releases/download/200k/vilt_200k_mlm_itm.ckpt).

#### Fine-tuned models

For the paper experiments, the models are fine-tuned from pre-trained models on tasks VQA and NLVR2.
UNITER and LXMERT detail their instructions in their repositories.

Once LXMERT finetuned the you will have two folders ```nlvr2_lxr955``` and ```vqa_lxr955``` .
those folders must be copied into ```lxmert/snap/pretrained/$FOLDER```. We used the BEST.pth checkpoint for both VQA and NLVR2 

For UNITER you will have also two folders for VQA and NLVR2, they must be copied into ```UNITER/pretrained/$FOLDER```.
We used ```model_step_6500.pt``` for NLVR2 and ```model_step_6000.pt``` for VQA.

We use the fine-tuned ViLT models made available by authors in our experiments: [VQA](https://github.com/dandelin/ViLT/releases/download/200k/vilt_vqa.ckpt) and [NLVR2](https://github.com/dandelin/ViLT/releases/download/200k/vilt_nlvr2.ckpt).

### Computing the image region representations

We use [MS-COCO](https://cocodataset.org/#home), [Flower-102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) and [Flickr30k](http://shannon.cs.illinois.edu/DenotationGraph/) using https://www.kaggle.com/hsankesara/flickr-image-dataset images.
UNITER and LXMERT both use image region representations as input of their model.
As a result, we compute image region representations for the probing tasks datasets.

- For UNITER, we compute the representations using the instructions in their repository and save them in ```UNITER/uniter_data/images_data``` in their respective folders.

- For LXMERT, we compute the representations using the instructions in their repository and save them in ```lxmert/data``` in their respective folders.

- For ViLT, we use the available code to build the .arrows files following ViLT protocol and save them in the ```data``` folder.

### Computing the representations from scratch 
Using the models, data and image region representations, we compute the representations used for the probing tasks.

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
./representations_scripts/get_repr_coco.sh # for the object counting task  
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
./run/get_repr_coco.sh # for the object counting task  
```

3 - For ViLT, execute:
```sh
cd ViLT
python run.py with data_root=data/probing num_gpus=1 num_nodes=1 per_gpu_batchsize=32 task_probing test_only=True load_path="checkpoints/$MODEL" # For all tasks
```

4 - For the baselines models (BERT, VIT and RESNET), execute :
```sh
cd probing_tasks
./get_vit_representation.sh
./get_bert_representation.sh
./get_resnet_representation.sh
```
At the end of this step, all representations will be saved in the representations directory. 

You can skip this steps by downloading representations (available on demand).

### Reproducing the probing experiments
For UNITER and LXMERT: 
After having computed the representations, reproduce the experiments on linear probing models by executing those commands : 
```sh
cd probing_tasks
./altercaps.sh # for the adversarial caption task 
./bshift.sh  # for the bigramshift task 
./colors_cls.sh  # for the color classification task  
./flowers.sh  # for the flower classification task  
./postag.sh  # for the pos-tagging task  
./objCount.sh  # for the object counting task
./pos_size.sh  # for the position and  size tasks 
```

For ViLT:
After having computed the representations, reproduce the experiments on linear probing models by executing those commands :
```sh
cd ViLT
./run_probing.sh # for the pre-trained model experiments
./run_probing_vqa.sh # for the VQA fine-tuned experiments
./run_probing_nlvr.sh # for the NLVR fine-tuned experiments


