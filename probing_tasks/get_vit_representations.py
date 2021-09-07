import torch
import csv
import pickle
import argparse
import torch
from tqdm import tqdm
from utils import *
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image


feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')



def get_representation(im,model,feature_extractor,device):
    image = Image.open(im).convert('RGB')

    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(inputs['pixel_values'].to(device))
    last_hidden_states = outputs.last_hidden_state
    return last_hidden_states , last_hidden_states[:,0,:]











def main(args):
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

            with torch.no_grad():
                feat,cls_ = get_representation(im_path,model,feature_extractor,device)
            fnames.append(line[0])
            representations.append(cls_.to("cpu"))
            targets.append(line[-1])
            del feat
            del cls_

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

                    
    args = parser.parse_args()
    main(args)

