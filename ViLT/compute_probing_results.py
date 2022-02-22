from probing_utils import *
import argparse
import pickle
from collections import Counter
import numpy as np
from tqdm import tqdm
import os 
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from  sklearn.metrics import mean_squared_error as mse
import random

def process(data,fnames,model, target2id, id2fname, fnames_train, coco=False, objcount=False):
    train_input=[]
    train_targets = []
    val_input=[]
    val_targets = []
    if model in ['vilt']:
        a = 'raw_cls_feats'
    for el in data:
        idx = el['raw_index'][0]
        if coco:
            num_id = str(int(id2fname[idx][14:-4]))
            test = num_id not in fnames
        elif fnames_train :
            test = id2fname[idx] in fnames 
        else:
            test = id2fname[idx][:-4]  not in fnames
        if not test:    
            val_input.append(el[a].float())
            if objcount is True:
                val_targets.append(int(el['target'][0]))
            else:
                val_targets.append(target2id[el['target'][0]])
        else:
            train_input.append(el[a].float())
            if objcount is True:
                train_targets.append(int(el['target'][0]))
            else:
                train_targets.append(target2id[el['target'][0]])
    
    print(len(train_input))
    print(len(val_input))
    print(len(train_targets))
    print(len(val_targets))


    input_train = torch.cat(train_input)
    input_val =torch.cat(val_input)  
    target_train = torch.tensor(train_targets)
    target_val = torch.tensor(val_targets)

    print(model+"_input_train",input_train.shape)
    print(model+"_input_val", input_val.shape)
    print(model+"_target_train",target_train.shape)
    print(model+"_target_val",target_val.shape)
    print()
    return input_train, input_val, target_train, target_val



def main(args):
    random.seed(0)

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODELS = True
    fnames_train,coco = True,False
    if args.task in ['pos', 'size']:
        with open('../probing-vl-models/probing_tasks/split/train_'+args.task+'_fnames.txt') as f:
            fnames = f.read().splitlines()
        csv_data = pd.read_csv('../probing-vl-models/datasets/flickr_3000_'+args.task+'.csv', sep='|')
    elif args.task in ['pos_dec', 'size_dec']:
        with open('../probing-vl-models/probing_tasks/split/train_fnames_'+args.task+'.txt') as f:
            fnames = f.read().splitlines()
        csv_data = pd.read_csv('../probing-vl-models/datasets/flickr_3000_'+args.task+'.csv', sep='|')
    elif args.task in ['colors']:
        with open('../probing-vl-models/probing_tasks/split/'+args.task+'_train_fnames.txt') as f:
            fnames = f.read().splitlines()
        csv_data = pd.read_csv('../probing-vl-models/datasets/flickr_3000_'+args.task+'.csv', sep='|')
    elif args.task in ['colors_dec']:
        with open('../probing-vl-models/probing_tasks/split/colors_train_fnames_dec.txt') as f:
            fnames = f.read().splitlines()
        csv_data = pd.read_csv('../probing-vl-models/datasets/flickr_3000_'+args.task+'.csv', sep='|')
    elif args.task in ['bshift']:
        with open('../probing-vl-models/probing_tasks/split/'+args.task+'_test.txt') as f:
            fnames = f.read().splitlines()
        csv_data = pd.read_csv('../probing-vl-models/datasets/flickr_3000_'+args.task+'.csv', sep='|')
        fnames_train = False
    elif args.task in ['bshift_dec']:
        with open('../probing-vl-models/probing_tasks/split/bshift_test_dec.txt') as f:
            fnames = f.read().splitlines()
        csv_data = pd.read_csv('../probing-vl-models/datasets/flickr_3000_'+args.task+'.csv', sep='|')
        fnames_train = False
    elif args.task in ['objcount']:
        with open('../probing-vl-models/probing_tasks/split/coco_val.txt') as f:
            fnames = f.read().splitlines()
        csv_data = pd.read_csv('../probing-vl-models/datasets/coco_uniter.csv', sep='|')
        fnames_train,coco = False,True
    elif args.task in ['objcount_dec']:
        with open('../probing-vl-models/probing_tasks/split/coco_val_dec.txt') as f:
            fnames = f.read().splitlines()
        csv_data = pd.read_csv('../probing-vl-models/datasets/coco_uniter_dec.csv', sep='|')
        fnames_train,coco = False,True
    elif args.task in ['altcap']:
        with open('../probing-vl-models/probing_tasks/split/val_altercaps.txt') as f: 
            fnames = f.read().splitlines()
        csv_data1 = pd.read_csv('../probing-vl-models/datasets/coco_altercap_uniter_train.csv', sep='|') 
        csv_data2 = pd.read_csv('../probing-vl-models/datasets/coco_altercap_uniter_val.csv', sep='|') 
        csv_data = pd.concat([csv_data1, csv_data2], ignore_index=True)
        fnames_train,coco = False,True

    elif args.task in ['altcap_dec']:
        with open('../probing-vl-models/probing_tasks/split/val_uniter_altercaps_dec.txt') as f: 
            fnames = f.read().splitlines()
        csv_data1 = pd.read_csv('../probing-vl-models/datasets/coco_altercap_uniter_train_dec.csv', sep='|') 
        csv_data2 = pd.read_csv('../probing-vl-models/datasets/coco_altercap_uniter_val_dec.csv', sep='|') 
        csv_data = pd.concat([csv_data1, csv_data2], ignore_index=True)
        fnames_train,coco = False,True
    

    print('csv', csv_data.shape)
    print("Vilt")
    
    with open(args.vilt_data, 'rb') as handle:
        vilt_data = pickle.load(handle)

    target2id = {}
    id2target = {}
    id2fname = {}
    
    all_targets = []
    for el in vilt_data:
        all_targets.append(el['target'][0])
   
    vilt_data_start_idx = vilt_data[0]['raw_index'][0]
    for i, row in csv_data.iterrows():
        id2fname[i+vilt_data_start_idx] = row['image_name']
    
    for i,c in enumerate(sorted(list(set(all_targets)))):
        target2id[c] = i
        id2target[i] = c


    input_train, input_val, target_train, target_val = process(vilt_data,fnames, 'vilt', target2id, id2fname, fnames_train, coco, objcount=('objcount' in args.task))
    

    print()
    print("target_train",np.mean(target_train.numpy()))
    print("target_val",np.mean(target_val.numpy()))
    print()
    
    bs = args.batch_size
    infos={}
    nb_epochs = args.epochs
    for seed in [60,43,20,12,34]:
        torch.manual_seed(seed)
        loader_train = data_loaders_manueal_split(input_train, target_train,bs,True,seed)
        loader_val = data_loaders_manueal_split(input_val, target_val,bs,True,seed)

        loaders = [(loader_train, loader_val)]
        models = ['VILT']
        
        if 'objcount' in args.task:
            print('obj counting')
            lin = Net(768,1)
            mlp_model = MLP2(768,384,1,torch.nn.ReLU)
            criterion = torch.nn.MSELoss()
            infos[seed] = run(loaders,models,lin,mlp_model,nb_epochs,device,criterion,seed,mlp_use=True,display=True)

        else:
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
    parser.add_argument("--task",
                        type=str, required=True ,choices =['pos','size','colors','bshift', 'pos_dec', 
                        'size_dec', 'colors_dec', 'bshift_dec', 'objcount', 'objcount_dec', 'altcap', 'altcap_dec'],
                        help="The probing_task.")
    parser.add_argument("--batch_size",
                        type=int, default=16,
                        help="The input train corpus.")
    parser.add_argument("--epochs",
                        type=int, default=30,
                        help="probing nb epochs.")


    parser.add_argument("--vilt_data",
                        type=str, required=True,
                        help="vilt representations pickle")


    
    args = parser.parse_args()

    print(args)
    infos = main(args)
    print("saving ....")
    with open('probing_results/'+args.exp_name+'.pickle', 'wb') as handle:
        pickle.dump(infos, handle, protocol=pickle.HIGHEST_PROTOCOL)
