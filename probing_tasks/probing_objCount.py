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
import random

def process(uniter_data,fnames,model,color2id):
    uniter_train_input=[]
    uniter_train_targets = []
    uniter_val_input=[]
    uniter_val_targets = []
    if model in ['bert','vit','resnet','uniter']:
        a= 'representations'
    else:
        a = "pooled_representation"
    for rep,fn,tar in zip(uniter_data[a],uniter_data["fnames"],uniter_data["targets"]):
        if model == 'uniter':
            test = get_id(fn)  in fnames 
        if model in  ['bert','vit','resnet']:
            test = get_id(fn[:-4]+'.npz') in fnames 
        elif model =='lxmert':
            test= get_id(fn+'.npz') in fnames
        if test:
            if model == 'resnet':
                uniter_val_input.append(rep.unsqueeze(0))
            elif model in ['uniter','bert']:
                uniter_val_input.append(rep[:,0,:])
            else : 
                uniter_val_input.append(rep)

            uniter_val_targets.append(int(tar))
        else:
            if model == 'resnet':
                uniter_train_input.append(rep.unsqueeze(0))
            elif model in ['uniter','bert']:
                uniter_train_input.append(rep[:,0,:])
            else : 
                uniter_train_input.append(rep)
            uniter_train_targets.append(int(tar))


    print(len(uniter_train_input))
    print(len(uniter_val_input)  )
    print(len(uniter_train_targets))
    print(len(uniter_val_targets))

    uniter_input_train = torch.cat(uniter_train_input)
    uniter_input_val =torch.cat(uniter_val_input)  
    uniter_target_train = torch.tensor(uniter_train_targets)
    uniter_target_val = torch.tensor(uniter_val_targets)

    print(model+"_input_train",uniter_input_train.shape)
    print(model+"_input_val",uniter_input_val.shape)
    print(model+"_target_train",uniter_target_train.shape)
    print(model+"_target_val",uniter_target_val.shape)
    print()
    return uniter_input_train,uniter_input_val,uniter_target_train,uniter_target_val



def main(args):
    random.seed(0)

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODELS = True
    if 'dec' in args.exp_name or 'vqa' in args.exp_name or 'nlvr' in args.exp_name:
        MODELS = False

    if 'dec' in args.exp_name:
     with open('./split/coco_val_dec.txt') as f:
        fnames = f.read().splitlines()
    else:
     with open('./split/coco_val.txt') as f:
        fnames = f.read().splitlines()
    print("LXMERT")
    with open(args.lxmert_data, 'rb') as handle:
            lxmert_data = pickle.load(handle)
    print(lxmert_data['fnames'][:10])
    print(lxmert_data['targets'][:10])
    print(len(lxmert_data['targets']))
    print(len(lxmert_data['fnames']))



    with open(args.uniter_data, 'rb') as handle:
            uniter_data = pickle.load(handle)    
    print(uniter_data['fnames'][:10])
    print(uniter_data['targets'][:10])

    if MODELS:
        with open(args.bert_data, 'rb') as handle:
           bert_data = pickle.load(handle)
        print(bert_data.keys())
        print(bert_data['fnames'][:10])
        print(bert_data['targets'][:10])


        with open(args.resnet_data, 'rb') as handle:
                resnet_data = pickle.load(handle)
        print(resnet_data.keys())
        print(resnet_data['fnames'][:10])
        print(resnet_data['targets'][:10])

        with open(args.vit_data, 'rb') as handle:
                vit_data = pickle.load(handle) 
        print(vit_data.keys())
        print(vit_data['fnames'][:10])
        print(vit_data['targets'][:10])


    
    color2id = {}
    id2color = {}
    for i,c in enumerate(sorted(list(set(uniter_data['targets'])))):
        color2id[c] = i
        id2color[i] = c

    if 'dec' in args.exp_name :# if dec we have train file names , if not dec we have val fnames
        uniter_input_val,uniter_input_train,uniter_target_val,uniter_target_train = process(uniter_data,fnames,'uniter',color2id)
        lxmert_input_val,lxmert_input_train,lxmert_target_val,lxmert_target_train = process(lxmert_data,fnames,'lxmert',color2id)
    else : 
        uniter_input_train,uniter_input_val,uniter_target_train,uniter_target_val = process(uniter_data,fnames,'uniter',color2id)
        lxmert_input_train,lxmert_input_val,lxmert_target_train,lxmert_target_val = process(lxmert_data,fnames,'lxmert',color2id)
    if MODELS :
        bert_input_train,bert_input_val,bert_target_train,bert_target_val = process(bert_data,fnames,'bert',color2id)

        vit_input_train,vit_input_val,vit_target_train,vit_target_val = process(vit_data,fnames,'vit',color2id)

        resnet_input_train,resnet_input_val,resnet_target_train,resnet_target_val = process(resnet_data,fnames,'resnet',color2id)


    




    print()
    print("uniter_target_train",np.mean(uniter_target_train.numpy()))
    print("uniter_target_val",np.mean(uniter_target_val.numpy()))
    print()
    print("lxmert_target_train",np.mean(lxmert_target_train.numpy()))
    print("lxmert_target_val",np.mean(lxmert_target_val.numpy()))
    print()
    if MODELS : 
        print("bert_target_train",np.mean(bert_target_train.numpy()))
        print("bert_target_val",np.mean(bert_target_val.numpy()))
        print()
        print()
        print("vit_target_train",np.mean(vit_target_train.numpy()))
        print("vit_target_val",np.mean(vit_target_val.numpy()))
        print()

        print()
        print("resnet_target_train",np.mean(resnet_target_train.numpy()))
        print("resnet_target_val",np.mean(resnet_target_val.numpy()))
        print()

    bs = args.batch_size
    infos={}
    nb_epochs = args.epochs
    for seed in [60,43,20,12,34]:
        torch.manual_seed(seed)
        uniter_loader_train = data_loaders_manueal_split(uniter_input_train,uniter_target_train,bs,True,seed)
        uniter_loader_val = data_loaders_manueal_split(uniter_input_val,uniter_target_val,bs,True,seed)

        lxmert_loader_train = data_loaders_manueal_split(lxmert_input_train,lxmert_target_train,bs,True,seed)
        lxmert_loader_val = data_loaders_manueal_split(lxmert_input_val,lxmert_target_val,bs,True,seed)



        if MODELS :  
            bert_loader_train = data_loaders_manueal_split(bert_input_train,bert_target_train,bs,True,seed)
            bert_loader_val = data_loaders_manueal_split(bert_input_val,bert_target_val,bs,True,seed)

            vit_loader_train = data_loaders_manueal_split(vit_input_train,vit_target_train,bs,True,seed)
            vit_loader_val = data_loaders_manueal_split(vit_input_val,vit_target_val,bs,True,seed)

            resnet_loader_train = data_loaders_manueal_split(resnet_input_train,resnet_target_train,bs,True,seed)
            resnet_loader_val = data_loaders_manueal_split(resnet_input_val,resnet_target_val,bs,True,seed)


            loaders = [(uniter_loader_train,uniter_loader_val),(lxmert_loader_train,lxmert_loader_val),(bert_loader_train,bert_loader_val),(vit_loader_train,vit_loader_val),(resnet_loader_train,resnet_loader_val)]
            models = ['UNITER','LXMERT','BERT','VIT','RESNET'] 
        else : 
            loaders = [(uniter_loader_train,uniter_loader_val),(lxmert_loader_train,lxmert_loader_val)]
            models = ['UNITER','LXMERT']


        lin = Net(768,1)
        mlp_model = MLP2(768,384,1,torch.nn.ReLU)
        criterion = torch.nn.MSELoss()
        infos[seed] = run(loaders,models,lin,mlp_model,nb_epochs,device,criterion,seed,mlp_use=True,display=True)

    return infos




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--exp_name",
                        type=str, required=True,
                        help="The input train corpus.")
    parser.add_argument("--batch_size",
                        type=int, default=16,
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


    parser.add_argument("--bert_data",
                        type=str, required=True,
                        help="bert representations pickle")

    parser.add_argument("--resnet_data",
                        type=str, required=True,
                        help="bert representations pickle")
    parser.add_argument("--vit_data",
                        type=str, required=True,
                        help="bert representations pickle")


    
    args = parser.parse_args()

    print(args)
    infos = main(args)
    print("saving ....")
    with open('probing_results/'+args.exp_name+'.pickle', 'wb') as handle:
        pickle.dump(infos, handle, protocol=pickle.HIGHEST_PROTOCOL)
