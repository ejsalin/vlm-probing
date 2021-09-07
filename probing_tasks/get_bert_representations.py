from utils import *
import argparse
import pickle
from collections import Counter
import csv
import os
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.to('cuda')



def Bert_inf(model,tokenizer,text):
  tokenized_text = tokenizer.tokenize(text)

  indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
  tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
  with torch.no_grad():
    # See the models docstrings for the detail of the inputs
    outputs = model(tokens_tensor)
  return (outputs[0]) , tokenized_text

def main(args):
    print(args)
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # open representations

    with open(args.ann) as csvfile:
        ann = csv.reader(csvfile, delimiter='|', quotechar='|',)
        next(ann)

        bert_input = []
        bert_targets = []
        bert_txt = []
        fnames = []
        for line in tqdm(ann):
              rep,txt_tok = Bert_inf(model,tokenizer,'[CLS] '+line[2]+' [SEP]')

              if args.ann.endswith("colors.csv"):
                
                #print(rep.shape," ",len(txt_tok) )
                #print(txt_tok)
                idx = [i for i,x in enumerate(txt_tok) if x == '[MASK]'][0]
                #print(idx)
                bert_input.append(rep[:,idx,:])
                bert_targets.append(line[-1])
                bert_txt.append(txt_tok)
                fnames.append(line[0])

              else :  
                bert_input.append(rep)
                bert_targets.append(line[-1])
                bert_txt.append(txt_tok)
                fnames.append(line[0])

  
  
    bert_data = {}
    bert_data['texts'] =bert_txt
    bert_data['targets'] = bert_targets
    bert_data['representations'] = bert_input
    bert_data['fnames'] = fnames
    with open(args.output,'wb') as handle:
          pickle.dump(bert_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

 
 









if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters

    parser.add_argument("--ann",
                        type=str, required=True,
                        help="uniter representations pickle")
    parser.add_argument("--output",
                        type=str, required=True,
                        help="uniter representations pickle")
    
    args = parser.parse_args()

    main(args)

