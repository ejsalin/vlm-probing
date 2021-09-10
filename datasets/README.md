## Datasets for probing tasks experiments

This folder groups the datasets used for probing tasks experiments.
Details on these experiments are provided in the paper.

For each ```.csv``` file:

- ```image_name``` indicates the image of the instance

- ```comment_number```indicates the number of the caption associated to the image, as some datasets have several captions for each image.

- ```comment```indicates the text input used for the model

- ```original``` indicates the original caption before eventual processing due to the task.

- ```target```indicates the target of the task


### Language probing tasks

These are based on Flick30k.

#### Bigram shift

The dataset used for the bigram shift task is ```flickr_3000_bshift.csv```, and its mismatched images equivalent is ```flickr_3000_bshift_dec.csv```.

#### POS tagging

The dataset used for the POS tagging task is ```flickr_3000.csv```, and its mismatched images equivalent is ```flickr_3000_dec.csv```.
The target used is computed using ```probing-vl-models/probing_tasks/probing_postag.py```

### Vision probing tasks

#### Object counting

This task is based on MS-COCO.
The dataset used for the Object counting task are ```coco_lxmert.csv``` and ```coco_uniter.csv```, and their mismatched caption equivalent is ```coco_lxmert_dec.csv``` and ```coco_uniter_dec.csv```.
We make sure that MS-COCO images were not used for the training phase of UNITER and LXMERT pre-training.
As UNITER and LXMERT do not use the same MS-COCO subsets for their pre-training, UNITER and LXMERT models are not trained on the same datasets for the probing task, but they are evaluated on the same dataset.

#### Fine-grained classification

This task is based on Flower-102.
The dataset used for the Fine-grained task is ```flower_data.csv```.

### Multimodal probing tasks
We present new multimodal tasks by modifying captions (see paper).

#### Color
This task is based on Flickr30k dataset.
The dataset used for the Color task is ```flickr_3000_colors.csv```, and its mismatched images equivalent is ```flickr_3000_colors_dec.csv```.

#### Size
This task is based on Flickr30k dataset.
The dataset used for the Size task is ```flickr_3000_size.csv```, and its mismatched images equivalent is ```flickr_3000_size_dec.csv```.

#### Position
This task is based on Flickr30k dataset.
The dataset used for the Position task is ```flickr_3000_pos.csv```, and its mismatched images equivalent is ```flickr_3000_pos_dec.csv```.

#### Adversarial captions
This task is based on MS-COCO dataset.
The datasets used for the Adversarial captions task are ```coco_altercap_lxmert.csv```, and its mismatched images equivalent is ```_dec.csv```.
