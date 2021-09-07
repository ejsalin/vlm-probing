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
import spacy 
import random
nlp = spacy.load("en_core_web_sm")

def get_uniter_words(data_uniter):
    uniter_texts_ = [tokenToString(text) for text in data_uniter["texts"]]

    uniter_words=[]
    uniter_repr,uniter_texts = [],[]
    for rpr ,ids,text in tqdm(zip(data_uniter['representations'],data_uniter['text_ids'],data_uniter['texts']),total=len(uniter_texts_)):
            a = rpr[:,:ids.shape[1]]
            uniter_texts.append(text)
            uniter_repr.append(a)

    for text , rep in tqdm(zip(uniter_texts,uniter_repr),total= len(uniter_repr)):
      #print("aaaaaa")
      assert len(text.split(" "))==rep.shape[1]
      aa = 0
      t = tokenToString(text)
      doc = nlp(t)
      tokens = text.split(" ")[1:-1]
      r = rep[:,1:-1,:] 

      for j in range(len(doc)):
        #print("j = ",j)
        for i in range(j+aa,len(tokens)):
          #print("i = ",i)
          if not tokens[i].startswith("##"):
            if tokens[i].lower() != doc[j].text.lower() and not doc[j].text.lower().startswith(tokens[i].lower()):
                print("UNITER ",tokens[i]," ",doc[j].text)
            else:
                uniter_words.append((tokens[i],doc[j].tag_,r[:,i]))
                
            break
          else:
            #print(tokens[j])
            aa+=1   
    print("uniter",len(uniter_words))
    return uniter_words

def get_lxmert_words(data_lxmert):
    lxmert_texts = []
    lxmert_repr=[]
    lxmert_words= []
    lxmert_texts_ = [tokenToString(text) for text in data_lxmert["texts"]]

    for rep,text in tqdm(zip(data_lxmert['lang'],data_lxmert['texts']),total=len(lxmert_texts_)):
            lxmert_repr.append(rep)
            lxmert_texts.append(text)

    for text , rep in tqdm(zip(lxmert_texts,lxmert_repr),total= len(lxmert_repr)):
      
      assert len(text.split(" "))==rep.shape[1]
      aa = 0
      t = tokenToString(text)
      doc = nlp(t)
      tokens = del_PAD(text).split(" ")[1:-1]
      r = rep[:,1:-1,:]
      for j in range(len(doc)):
        #print("j = ",j)
        for i in range(j+aa,len(tokens)):
          #print("i = ",i)
          if not tokens[i].startswith("##"):
            if tokens[i].lower() != doc[j].text.lower() and not doc[j].text.lower().startswith(tokens[i].lower()):
                print("LXMERT ",tokens[i]," ",doc[j].text)
            else:
                lxmert_words.append((tokens[i],doc[j].tag_,r[:,i]))
            break
          else:
            #print(tokens[j])
            aa+=1
    print("lxmert",len(lxmert_words))
    return lxmert_words

def get_bert_words(data_bert):
    #bert_texts_ = [tokenToString(text) for text in data_bert["texts"]]

    bert_repr,bert_texts,bert_words = [],[],[]
    for rep,texts in tqdm(zip(data_bert['representations'],data_bert['texts']),total=len(data_bert['texts'])):
            bert_repr.append(rep)
            bert_texts.append(texts)
    for text , rep , in tqdm(zip(bert_texts,bert_repr),total= len(bert_repr)):
      text = " ".join(text)

      assert len(del_PAD(text).split(" "))==rep.shape[1]
      aa = 0
      t = tokenToString(text)
      doc = nlp(t)
      tokens = del_PAD(text).split(" ")[1:-1]
      r = rep[:,1:-1,:]
      for j in range(len(doc)):
        #print("j = ",j)
        for i in range(j+aa,len(tokens)):
          #print("i = ",i)
          if not tokens[i].startswith("##"):
            if tokens[i].lower() != doc[j].text.lower() and not doc[j].text.lower().startswith(tokens[i].lower()):
                print("BERT ",tokens[i]," ",doc[j].text)
            else:
                bert_words.append((tokens[i],doc[j].tag_,r[:,i]))
            break
          else:
            #print(tokens[j])
            aa+=1

    return bert_words


def main(args):
    random.seed(0)
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODELS = True
    if 'dec' in args.exp_name or 'vqa' in args.exp_name or 'nlvr' in args.exp_name:
        MODELS = False
    with open(args.lxmert_data, 'rb') as handle:
            lxmert_data = pickle.load(handle)
    print(lxmert_data['fnames'][:3])
    with open(args.uniter_data, 'rb') as handle:
            uniter_data = pickle.load(handle)  
    print(uniter_data['fnames'][:3])
    with open(args.bert_data, 'rb') as handle:
            bert_data = pickle.load(handle)    
    print(bert_data['fnames'][:3])



    if MODELS:
        b = set([get_id(f) for f in uniter_data['fnames']])&set(lxmert_data['fnames'])&set([f[:-4] for f in bert_data['fnames']])
    else : 
        b = set([get_id(f) for f in uniter_data['fnames']])&set(lxmert_data['fnames'])
    print(len(b))
    
    fnames_comm = random.sample(b,1000)



    print('uniter_data  train processing ...')
    for example,title in zip([fnames_comm],['uniter_train']):
      fnames, texts, representations, pooled_representation, text_ids, gather_ids, targets = [],[],[],[],[],[],[]      
      added_fnames = []

      for fnames_, texts_, representations_, pooled_representation_, text_ids_, gather_ids_, targets_  in tqdm(zip(uniter_data['fnames'],uniter_data['texts'],uniter_data['representations'],uniter_data['pooled_representation'],uniter_data['text_ids'],uniter_data['gather_ids'],uniter_data['targets']),total=len(uniter_data['targets'])):
            if (get_id(fnames_) not in example) :
              fnames.append(fnames_) 
              texts.append(texts_) 
              representations.append(representations_) 
              targets.append(targets_)
              text_ids.append(text_ids_)
      print(len(texts))
      data_uniter_train = {'fnames':fnames, 'texts':texts, 'representations':representations, 'text_ids':text_ids,'targets':targets}
 

    print('uniter_data val processing ...')
    for example,title in zip([fnames_comm],['uniter_train']):
      fnames, texts, representations, pooled_representation, text_ids, gather_ids, targets = [],[],[],[],[],[],[]      
      added_fnames = []
      for fnames_, texts_, representations_, pooled_representation_, text_ids_, gather_ids_, targets_  in tqdm(zip(uniter_data['fnames'],uniter_data['texts'],uniter_data['representations'],uniter_data['pooled_representation'],uniter_data['text_ids'],uniter_data['gather_ids'],uniter_data['targets']),total=len(uniter_data['targets'])):

            if (get_id(fnames_)  in example) :
              fnames.append(fnames_) 
              texts.append(texts_) 
              representations.append(representations_) 
              targets.append(targets_)
              text_ids.append(text_ids_) 
      print(len(texts))     
      data_uniter_val = {'fnames':fnames, 'texts':texts, 'representations':representations, 'text_ids':text_ids,'targets':targets}
 



  # LXMERT DATA

    print('lxmert_data train processing ...')
    for example,title in zip([fnames_comm],['lxmert_train']):
      fnames, texts,visual, pooled_representation, lang, input_ids, segment_ids, input_mask, targets = [],[],[],[],[],[],[],[],[]
      added_fnames = []
      for fnames_, texts_, lang_,  in tqdm(zip(lxmert_data['fnames'], lxmert_data['texts'] , lxmert_data['lang']),total=len(lxmert_data['lang'])):
 
            if fnames_ not in example:
              texts.append(texts_)
              lang.append(lang_) 
      print(len(texts))
      data_lxmert_train = { 'texts':texts, 'lang':lang}
      
    print('lxmert_data val processing ...')
    for example,title in zip([fnames_comm],['lxmert_train']):
      fnames, texts,visual, pooled_representation, lang, input_ids, segment_ids, input_mask, targets = [],[],[],[],[],[],[],[],[]
      added_fnames = []
      for fnames_, texts_, lang_,  in tqdm(zip(lxmert_data['fnames'], lxmert_data['texts'] , lxmert_data['lang']),total=len(lxmert_data['lang'])):
            if fnames_  in example:
              texts.append(texts_)
              lang.append(lang_)
      print(len(texts)) 
      data_lxmert_val = { 'texts':texts, 'lang':lang}

    if MODELS:
        print('bert_data train processing ...')
        for example,title in zip([fnames_comm],['uniter_train']):
          fnames, texts, representations, pooled_representation, text_ids, gather_ids, targets = [],[],[],[],[],[],[]      
          added_fnames = []
          for fnames_, texts_, representations_, targets_  in tqdm(zip(bert_data['fnames'],bert_data['texts'],bert_data['representations'],bert_data['targets']),total=len(bert_data['targets'])):
                if (fnames_[:-4]  not in example) :
                  fnames.append(fnames_) 
                  texts.append(texts_) 
                  representations.append(representations_) 
                  targets.append(targets_)
                  text_ids.append(text_ids_)
          print(len(texts))
          data_bert_train = {'fnames':fnames, 'texts':texts, 'representations':representations, 'text_ids':text_ids,'targets':targets}
    

        print('bert_data val processing ...')
        for example,title in zip([fnames_comm],['uniter_train']):
          fnames, texts, representations, targets = [],[],[],[]
          added_fnames = []
          for fnames_, texts_, representations_, targets_  in tqdm(zip(bert_data['fnames'],bert_data['texts'],bert_data['representations'],bert_data['targets']),total=len(bert_data['targets'])):
                if (fnames_[:-4]  in example) :
                  fnames.append(fnames_) 
                  texts.append(texts_) 
                  representations.append(representations_) 
                  targets.append(targets_)
                  text_ids.append(text_ids_)
          print(len(texts))      
          data_bert_val = {'fnames':fnames, 'texts':texts, 'representations':representations, 'text_ids':text_ids,'targets':targets}
 

 



        bert_words_train = get_bert_words(data_bert_train)
        bert_words_val = get_bert_words(data_bert_val)
        print(len(bert_words_train))
    uniter_words_train = get_uniter_words(data_uniter_train)
    uniter_words_val = get_uniter_words(data_uniter_val)
    print("uniter",len(uniter_words_train))
    print("uniter",len(uniter_words_val))
    lxmert_words_train = get_lxmert_words(data_lxmert_train)
    lxmert_words_val = get_lxmert_words(data_lxmert_val)
  

 



 




    label2id = {}
    l = set([x[1] for x in lxmert_words_train+lxmert_words_val+uniter_words_train+uniter_words_val])
    for i,l in enumerate(list(l)):
        label2id[l] = i



    lxmert_input_train = torch.cat([x[2] for x in lxmert_words_train],dim=0)
    lxmert_targets_train = torch.tensor([label2id[x[1]] for x in lxmert_words_train])
    lxmert_input_val = torch.cat([x[2] for x in lxmert_words_val],dim=0)
    lxmert_targets_val = torch.tensor([label2id[x[1]] for x in lxmert_words_val])

    print('lxmert_input_train',lxmert_input_train.shape)
    print('lxmert_targets_train',lxmert_targets_train.shape)
    print('lxmert_input_val',lxmert_input_val.shape)
    print('lxmert_targets_val',lxmert_targets_val.shape)

    if MODELS:
        bert_input_train = torch.cat([x[2] for x in bert_words_train],dim=0)
        bert_targets_train = torch.tensor([label2id[x[1]] for x in bert_words_train])
        bert_input_val = torch.cat([x[2] for x in bert_words_val],dim=0)
        bert_targets_val = torch.tensor([label2id[x[1]] for x in bert_words_val])

        print("bert_input_train",bert_input_train.shape)
        print("bert_targets_train",bert_targets_train.shape)
        print("bert_input_val",bert_input_val.shape)
        print("bert_targets_val",bert_targets_val.shape)

    print(len(uniter_words_train))
    uniter_input_train = torch.cat([x[2] for x in uniter_words_train],dim=0)
    uniter_targets_train = torch.tensor([label2id[x[1]] for x in uniter_words_train])
    uniter_input_val = torch.cat([x[2] for x in uniter_words_val],dim=0)
    uniter_targets_val = torch.tensor([label2id[x[1]] for x in uniter_words_val])
    print("uniter_input_train",uniter_input_train.shape)
    print("uniter_targets_train",uniter_targets_train.shape)
    print("uniter_input_val",uniter_input_val.shape)
    print("uniter_targets_val",uniter_targets_val.shape)



 

    bs = args.batch_size
    infos={}
    nb_epochs = args.epochs
    for seed in [60,43,20,0,12]:
        torch.manual_seed(seed)
        uniter_loader_train = data_loaders_manueal_split(uniter_input_train,uniter_targets_train,bs,True,seed)
        uniter_loader_val = data_loaders_manueal_split(uniter_input_val,uniter_targets_val,bs,True,seed)

        lxmert_loader_train = data_loaders_manueal_split(lxmert_input_train,lxmert_targets_train,bs,True,seed)
        lxmert_loader_val = data_loaders_manueal_split(lxmert_input_val,lxmert_targets_val,bs,True,seed)
        if MODELS:
            bert_loader_train = data_loaders_manueal_split(bert_input_train,bert_targets_train,bs,True,seed)
            bert_loader_val = data_loaders_manueal_split(bert_input_val,bert_targets_val,bs,True,seed)

        if MODELS:  
            loaders = [(uniter_loader_train,uniter_loader_val),(lxmert_loader_train,lxmert_loader_val),(bert_loader_train,bert_loader_val)]
            models = ['UNITER','LXMERT','BERT']
        else:
            loaders = [(uniter_loader_train,uniter_loader_val),(lxmert_loader_train,lxmert_loader_val)]
            models = ['UNITER','LXMERT']
        input_shape =loaders[0][0].dataset.__getitem__(2)[0].shape[0]
        output_shape = len(label2id)

        infos[seed],_,_ = run_cls(loaders,models,nb_epochs,device,seed,output_shape,mlp=True,display=False)

    return infos





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--exp_name",
                        type=str, required=True,
                        help="The input train corpus.")
    parser.add_argument("--batch_size",
                        type=int, default=256,
                        help="The input train corpus.")
    parser.add_argument("--epochs",
                        type=int, default=50,
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
