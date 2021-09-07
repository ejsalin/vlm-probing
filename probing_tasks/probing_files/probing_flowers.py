from utils import *
import argparse
import pickle
from collections import Counter
import numpy as np
from tqdm import tqdm
import os 
import numpy as np 
from sklearn.linear_model import LinearRegression
from  sklearn.metrics import mean_squared_error as mse
from collections import Counter

import scipy.io
mat = scipy.io.loadmat('./imagelabels.mat')
print(mat.keys())

img_tgt = {}
for i,t in enumerate(mat['labels'].tolist()[0]):
    fname = i+1
    #print(fname)
    img_tgt[fname] = t-1
print(list(img_tgt.keys())[-10:])
split = mat = scipy.io.loadmat('./setid.mat')
print(split['trnid'])


def main(args):
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    with open(args.uniter_data, 'rb') as handle:
            uniter_data = pickle.load(handle)

  
    



   
    data_uniter = {}
    for ex_fname,title in zip([split['trnid'][0],split['tstid'][0]],['train','val']):


      fnames, texts, representations, pooled_representation, text_ids, gather_ids, targets = [],[],[],[],[],[],[]
      
      added_fnames = []
      for fnames_, representations_, pooled_representation_  in tqdm(zip(uniter_data['fnames'],uniter_data['representations'],uniter_data['pooled_representation']),total=len(uniter_data['representations'])):
            fnames_  = fnames_[-8:-4]

            test = (int(fnames_) in ex_fname) and (fnames_ not in added_fnames)
           
            if test :

              fnames.append(fnames_) 
              representations.append(representations_) 
              pooled_representation.append(pooled_representation_) 
              targets.append(img_tgt[int(fnames_) ])

      print('fnames',len(fnames))
      print('representations',len(representations))
      print('pooled_representation',len(pooled_representation))

      
      data_uniter[title] = {'fnames':fnames,  'representations':representations, 'pooled_representation':pooled_representation,"targets":targets}
    
    del uniter_data

    with open(args.lxmert_data, 'rb') as handle:
            lxmert_data = pickle.load(handle)

  # LXMERT DATA
    data_lxmert = {}

    print('lxmert_data processing ...')
    for ex_fname,title in zip([split['trnid'][0],split['tstid'][0]],['train','val']):



      fnames, pooled_representation, nbobj = [],[],[]
      added_fnames = []


      for fnames_, pooled_representation_,nbobj_  in tqdm(zip(lxmert_data['fnames'], lxmert_data['pooled_representation'],  lxmert_data['nbobj']),total=len(lxmert_data['nbobj'])):

            fnames_  = fnames_[-5:]
            test = (int(fnames_) in ex_fname)  and (fnames_ not in added_fnames)
            if test :
              fnames.append(fnames_) 

              pooled_representation.append(pooled_representation_) 

              nbobj.append(img_tgt[int(fnames_)])

              added_fnames.append(fnames_)

      print('fnames',len(fnames))

      print('pooled_representation',len(pooled_representation))

      print('nbobj',len(nbobj))

      data_lxmert[title] = {'fnames':fnames, 'pooled_representation':pooled_representation,  'targets':nbobj}
      
    del lxmert_data

    with open(args.vit_data, 'rb') as handle:
            vit_data = pickle.load(handle)

  # vit DATA
    data_vit = {}
    print(vit_data.keys())
    print('vit_data processing ...')
    for ex_fname,title in zip([split['trnid'][0],split['tstid'][0]],['train','val']):



      fnames, pooled_representation, nbobj = [],[],[]
      added_fnames = []


      for fnames_, pooled_representation_,nbobj_  in tqdm(zip(vit_data['fnames'], vit_data['pooled_representations'],  vit_data['nobj']),total=len(vit_data['nobj'])):

            fnames_  = fnames_[-8:-4]
            test = (int(fnames_) in ex_fname)  and (fnames_ not in added_fnames)
            if test :
              fnames.append(fnames_) 

              pooled_representation.append(pooled_representation_.cpu()) 

              nbobj.append(img_tgt[int(fnames_)])

              added_fnames.append(fnames_)

      print('fnames',len(fnames))

      print('pooled_representation',len(pooled_representation))

      print('nbobj',len(nbobj))

      data_vit[title] = {'fnames':fnames, 'pooled_representations':pooled_representation,  'targets':nbobj}
      
    del vit_data
    with open(args.resnet_data, 'rb') as handle:
            resnet_data = pickle.load(handle)  
  # resnet DATA
    data_resnet = {}

    print('resnet_data processing ...')
    for ex_fname,title in zip([split['trnid'][0],split['tstid'][0]],['train','val']):



      fnames, pooled_representation, nbobj = [],[],[]
      added_fnames = []


      for fnames_, pooled_representation_,nbobj_  in tqdm(zip(resnet_data['fnames'], resnet_data['representations'],  resnet_data['nobj']),total=len(resnet_data['nobj'])):

            fnames_  = fnames_[-8:-4]
            test = (int(fnames_) in ex_fname)  and (fnames_ not in added_fnames)
            if test :
              fnames.append(fnames_) 

              pooled_representation.append(pooled_representation_.cpu()) 

              nbobj.append(img_tgt[int(fnames_)])

              added_fnames.append(fnames_)

      print('fnames',len(fnames))

      print('pooled_representation',len(pooled_representation))

      print('nbobj',len(nbobj))

      data_resnet[title] = {'fnames':fnames, 'representations':pooled_representation,  'targets':nbobj}
      
    del resnet_data







    print(data_lxmert.keys())
    lxmert_input_train = torch.cat([x for x in data_lxmert['val']['pooled_representation']],dim=0)
    lxmert_targets_train = torch.tensor(data_lxmert['val']['targets'])
    lxmert_input_val = torch.cat([x for x in data_lxmert['train']['pooled_representation']],dim=0)
    lxmert_targets_val = torch.tensor(data_lxmert['train']['targets'])

    print('lxmert_input_train',lxmert_input_train.shape)
    print('lxmert_targets_train',lxmert_targets_train.shape)
    print('lxmert_input_val',lxmert_input_val.shape)
    print('lxmert_targets_val',lxmert_targets_val.shape)


    uniter_input_train = torch.cat([x[:,0,:] for x in data_uniter['val']['representations']],dim=0)
    uniter_targets_train = torch.tensor(data_uniter['val']['targets'])
    uniter_input_val = torch.cat([x[:,0,:] for x in data_uniter['train']['representations']],dim=0)
    uniter_targets_val = torch.tensor(data_uniter['train']['targets'])

    print('uniter_input_train',uniter_input_train.shape)
    print('uniter_targets_train',uniter_targets_train.shape)
    print('uniter_input_val',uniter_input_val.shape)
    print('uniter_targets_val',uniter_targets_val.shape)

    vit_input_train = torch.cat([x for x in data_vit['val']['pooled_representations']],dim=0).to(device)
    vit_targets_train = torch.tensor(data_vit['val']['targets']).to(device)
    vit_input_val = torch.cat([x for x in data_vit['train']['pooled_representations']],dim=0).to(device)
    vit_targets_val = torch.tensor(data_vit['train']['targets']).to(device)

    print('vit_input_train',vit_input_train.shape)
    print('vit_targets_train',vit_targets_train.shape)
    print('vit_input_val',vit_input_val.shape)
    print('vit_targets_val',vit_targets_val.shape)

    resnet_input_train = torch.cat([x.unsqueeze(0) for x in data_resnet['val']['representations']],dim=0).to(device)
    resnet_targets_train = torch.tensor(data_resnet['val']['targets']).to(device)
    resnet_input_val = torch.cat([x.unsqueeze(0)  for x in data_resnet['train']['representations']],dim=0).to(device)
    resnet_targets_val = torch.tensor(data_resnet['train']['targets']).to(device)

    print('resnet_input_train',resnet_input_train.shape)
    print('resnet_targets_train',resnet_targets_train.shape)
    print('resnet_input_val',resnet_input_val.shape)
    print('resnet_targets_val',resnet_targets_val.shape)



 

    bs = args.batch_size
    infos={}
    nb_epochs = args.epochs
    for seed in [60,43,20,0,12]:
        torch.manual_seed(seed)
        uniter_loader_train = data_loaders_manueal_split(uniter_input_train,uniter_targets_train,bs,True,seed)
        uniter_loader_val = data_loaders_manueal_split(uniter_input_val,uniter_targets_val,bs,True,seed)

        lxmert_loader_train = data_loaders_manueal_split(lxmert_input_train,lxmert_targets_train,bs,True,seed)
        lxmert_loader_val = data_loaders_manueal_split(lxmert_input_val,lxmert_targets_val,bs,True,seed)

        vit_loader_train = data_loaders_manueal_split(vit_input_train,vit_targets_train,bs,True,seed)
        vit_loader_val = data_loaders_manueal_split(vit_input_val,vit_targets_val,bs,True,seed)

        resnet_loader_train = data_loaders_manueal_split(resnet_input_train,resnet_targets_train,bs,True,seed)
        resnet_loader_val = data_loaders_manueal_split(resnet_input_val,resnet_targets_val,bs,True,seed)

       
        loaders = [(uniter_loader_train,uniter_loader_val),(lxmert_loader_train,lxmert_loader_val),(vit_loader_train,vit_loader_val),(resnet_loader_train,resnet_loader_val)]
        models = ['UNITER','LXMERT','VIT','RESNET']

        output_shape = len(set(data_lxmert['train']['targets']+data_lxmert['val']['targets']))
        infos[seed],_,_ = run_cls(loaders,models,nb_epochs,device,seed,output_shape,mlp=True,display=False)

    return infos





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--exp_name",
                        type=str, required=True,
                        help="The input train corpus.")
    parser.add_argument("--batch_size",
                        type=int, default=56,
                        help="The input train corpus.")
    parser.add_argument("--epochs",
                        type=int, default=30,
                        help="probing nb epochs.")

    parser.add_argument("--lxmert_data",
                        type=str, required=True,
                        help="lxmert representations pickle")
    parser.add_argument("--uniter_data",
                        type=str, required=True,
                        help="uniter representations pickle")
    parser.add_argument("--resnet_data",
                        type=str, required=True,
                        help="lxmert representations pickle")
    parser.add_argument("--vit_data",
                        type=str, required=True,
                        help="uniter representations pickle")


    parser.add_argument("--dec",
                        type=int, choices=[1,0],required=True,
                        help="data ")
    parser.add_argument("--data",
                        type=str, choices=["flickr","coco"],required=True,
                        help="data ")
    
    args = parser.parse_args()

    print(args)
    infos = main(args)
    print(infos.keys())
    print("saving ....")
    with open('probing_results/'+args.exp_name+'.pickle', 'wb') as handle:
        pickle.dump(infos, handle, protocol=pickle.HIGHEST_PROTOCOL)





