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

The dataset used for the POS tagging task is ```.csv```, and its mismatched images equivalent is ```_dec.csv```.

### Vision probing tasks

#### Object counting

This task is based on MS-COCO.
The dataset used for the POS tagging task is ```.csv```, and its mismatched images equivalent is ```_dec.csv```.

#### Fine-grained classification

This task is based on Flower-102.
The dataset used for the POS tagging task is ```.csv```, and its mismatched images equivalent is ```_dec.csv```.

### Multimodal probing tasks
