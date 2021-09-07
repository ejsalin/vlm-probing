import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import torchvision

import torch
import pickle
from PIL import Image
import csv
import argparse

from os.path import exists
from time import time
from utils import *
import torch
from tqdm import tqdm




def converte_name(name):
    return 'COCO'+name[4:-3]+'jpg'

def get_representation(image,model,device,transform):
  image = transform(image)
  image = image.unsqueeze(0).to(device)

  out = model(image)
  return out

def main(args,model,transform):
    device = torch.device("cuda")  # support single GPU only
    model.eval().to(device)

    print("loading lxmert representations ...")
    if "coco" in args.ann:
        IMAGES_PATH = '../../DATA/COCOVAL/val2014/'
    if "flickr" in args.ann:
        IMAGES_PATH = '../../DATA/flickr30K/archive/flickr30k_images/flickr30k_images/'
    if "flower" in args.ann:
        IMAGES_PATH = '../jpg'
    representations = []
    targets = []
    fnames = []

    with open(args.ann) as csvfile:
        ann = csv.reader(csvfile, delimiter='|', quotechar='|',)
        next(ann)

        for line in tqdm(ann):
            if 'coco' in args.ann:
                im_path = IMAGES_PATH+line[0]
            else :
                im_path = IMAGES_PATH+line[0]

            image = Image.open(im_path).convert('RGB')
            feat = get_representation(image,model,device,transform).reshape([2048])
            fnames.append(line[0])
            representations.append(feat)
            targets.append(line[-1])


        with open(args.output, 'wb') as handle:
            pickle.dump({"fnames":fnames,"targets":targets,"representations":representations}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--ann",
                        type=str, required=True,
                        help="images dir path.")

    parser.add_argument("--output",
                        type=str, required=True,
                        help="images dir path.")

                            


    model = torchvision.models.resnet101(pretrained=True)
    modules=list(model.children())[:-1]
    model=torch.nn.Sequential(*modules)
    for p in model.parameters():
        p.requires_grad = False
    transform = transforms.Compose([
    transforms.Resize((224,224)),    
    transforms.ToTensor()
    
    ])


    args = parser.parse_args()
    main(args,model,transform)

