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


def main(args):
    random.seed(0)

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODELS = True

    print("LXMERT")
    with open(args.lxmert_data_train, 'rb') as handle:
            lxmert_data_train = pickle.load(handle)


    print("UNITER")
    with open(args.uniter_data_train, 'rb') as handle:
            uniter_data_train = pickle.load(handle)    

    print("LXMERT")
    with open(args.lxmert_data_val, 'rb') as handle:
            lxmert_data_val = pickle.load(handle)


    print("UNITER")
    with open(args.uniter_data_val, 'rb') as handle:
            uniter_data_val = pickle.load(handle)    


    if MODELS:
        print("BERT")
        with open(args.bert_data_train, 'rb') as handle:
           bert_data_train = pickle.load(handle)

        with open(args.bert_data_val, 'rb') as handle:
           bert_data_val = pickle.load(handle)




  






    target2id = {}
    id2target = {}
    for i,c in enumerate(sorted(list(set(uniter_data_train['targets'])))):
        target2id[c] = i
        id2target[i] = c




    
    uniter_input_train = torch.cat([rep[:,0,:] for rep in uniter_data_train["representations"]])
    uniter_input_val =torch.cat([rep[:,0,:] for rep in uniter_data_val["representations"]]) 
    uniter_target_train = torch.tensor([target2id[a] for a in uniter_data_train['targets']])
    uniter_target_val = torch.tensor([target2id[a] for a in uniter_data_val['targets']])

    bert_input_train = torch.cat([rep[:,0,:] for rep in bert_data_train["representations"]])
    bert_input_val =torch.cat([rep[:,0,:] for rep in bert_data_val["representations"]]) 
    bert_target_train = torch.tensor([target2id[a] for a in bert_data_train['targets']])
    bert_target_val = torch.tensor([target2id[a] for a in bert_data_val['targets']])

    lxmert_input_train = torch.cat([rep for rep in lxmert_data_train["pooled_representation"]])
    lxmert_input_val =torch.cat([rep for rep in lxmert_data_val["pooled_representation"]]) 
    lxmert_target_train = torch.tensor([target2id[a] for a in lxmert_data_train['targets']])
    lxmert_target_val = torch.tensor([target2id[a] for a in lxmert_data_val['targets']])



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

          

            loaders = [(uniter_loader_train,uniter_loader_val),(lxmert_loader_train,lxmert_loader_val),(bert_loader_train,bert_loader_val)]
            models = ['UNITER','LXMERT','BERT']
        else : 
            loaders = [(uniter_loader_train,uniter_loader_val),(lxmert_loader_train,lxmert_loader_val)]
            models = ['UNITER','LXMERT']
        input_shape =loaders[0][0].dataset.__getitem__(2)[0].shape[0]
        output_shape = len(target2id)

        infos[seed],_,_ = run_cls(loaders,models,nb_epochs,device,seed,output_shape,mlp=True,display=False)

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


    parser.add_argument("--lxmert_data_train",
                        type=str, required=True,
                        help="lxmert representations pickle")

    parser.add_argument("--uniter_data_train",
                        type=str, required=True,
                        help="uniter representations pickle")


    parser.add_argument("--bert_data_train",
                        type=str, required=True,
                        help="bert representations pickle")


    parser.add_argument("--lxmert_data_val",
                        type=str, required=True,
                        help="lxmert representations pickle")

    parser.add_argument("--uniter_data_val",
                        type=str, required=True,
                        help="uniter representations pickle")


    parser.add_argument("--bert_data_val",
                        type=str, required=True,
                        help="bert representations pickle")




    
    args = parser.parse_args()

    print(args)
    infos = main(args)
    print("saving ....")
    with open('probing_results/'+args.exp_name+'.pickle', 'wb') as handle:
        pickle.dump(infos, handle, protocol=pickle.HIGHEST_PROTOCOL)
