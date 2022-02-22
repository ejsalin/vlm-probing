from os import truncate
import matplotlib.pyplot as plt
import pickle
import torch 
import torch.nn as nn
import sklearn as sk
import numpy as np
import re
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from transformers import BertTokenizer
torch.manual_seed(0)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
import itertools
import numpy as np

def get_id(fname):
    if fname.isdecimal():
      return fname
    else:
      r1 = re.search(r'0[1-9][0-9]+.npz',fname)
      return r1.group(0)[1:-4]

def tokenToString(text):
  tokens = [token for token in text.split(" ") if token not in  ["[CLS]","[SEP]","[PAD]"]]
  toks =[]
  for i,t in enumerate(tokens):
    if t.startswith("##"):
      if toks != []:
        toks[-1] = toks[-1]+t[2:].lower()
    else:
      toks.append(t.lower())
  return " ".join(toks).lower()

def tokenToStringVilt(text):
  #print(text)
  #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  temp_tokens = tokenizer.convert_ids_to_tokens(text[0])
  #temp_text = tokens.joint(" ")
  tokens = [token for token in temp_tokens if token not in  ["[CLS]","[SEP]","[PAD]"]]
  toks =[]
  for i,t in enumerate(tokens):
    if t.startswith("##"):
      if toks != []:
        toks[-1] = toks[-1]+t[2:].lower()
    else:
      toks.append(t.lower())
  return " ".join(toks).lower()

def del_PAD(text):
  return " ".join([token for token in text.split(" ") if token !="[PAD]"])

class DataRepresentation(object):
  def __init__(self,inputs,targets):
    self.inputs = inputs
    self.targets = targets
  def __len__(self):
    return self.targets.shape[0]
  def __getitem__(self, index):
    y = self.targets[index]
    X = self.inputs[index]
    return X, y 

def data_loaders_manueal_split(inputs,targets,bs,shuffle,seed):
  torch.manual_seed(seed)
  dataset = DataRepresentation(inputs,targets)

  batch_size = bs
  shuffle_dataset = shuffle
  random_seed= 42


  dataset_size = len(dataset)
  indices = list(range(dataset_size))

  if shuffle_dataset :
      np.random.seed(random_seed)
      np.random.shuffle(indices)
      torch.manual_seed(random_seed)
  train_indices= indices

  # Creating PT data samplers and loaders:
  train_sampler = SubsetRandomSampler(train_indices)

  if shuffle is True:
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                             sampler=train_sampler)
    
  else:
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

  return train_loader 

def data_loaders(inputs,targets,bs,validation_split,shuffle,seed):
  dataset = DataRepresentation(inputs,targets)

  batch_size = bs
  validation_split = validation_split
  shuffle_dataset = shuffle
  random_seed= seed

  # Creating data indices for training and validation splits:
  dataset_size = len(dataset)
  indices = list(range(dataset_size))
  split = int(np.floor(validation_split * dataset_size))
  if shuffle_dataset :
      np.random.seed(random_seed)
      np.random.shuffle(indices)
  train_indices, val_indices = indices[split:], indices[:split]

  # Creating PT data samplers and loaders:
  train_sampler = SubsetRandomSampler(train_indices)
  valid_sampler = SubsetRandomSampler(val_indices)

  train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                             sampler=train_sampler)
  validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                 sampler=valid_sampler)
  return train_loader , validation_loader

class Net(nn.Module):
    def __init__(self, D_in,D_out):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, D_out)
        
    def forward(self, x):
        return self.linear1(x)

class MLP2(torch.nn.Module):
  def __init__(self,input_size,hidden_size,output_size,activation):
    super(MLP2, self).__init__()
    self.layer1 = torch.nn.Linear(input_size,hidden_size)
    self.layer2 = torch.nn.Linear(hidden_size,output_size)
    self.activation = activation()
  def forward(self,x):
    return self.layer2(self.activation(self.layer1(x)))



class AverageMeter(object):
  '''A handy class for moving averages''' 
  def __init__(self):
    self.reset()
  def reset(self):
    self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def train_loop(model,max_epochs,criterion,opt,train_loader,validation_loader,device,disp=True):
    model.to(device)

    train_loss , val_loss = [],[]
    for epoch in range(max_epochs):
        # Training
        losses = AverageMeter()
        val_losses = AverageMeter()
        for local_batch, local_labels in train_loader:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            opt.zero_grad()

            out = model(local_batch)

            loss = criterion(out , local_labels.float().unsqueeze(1))
            loss.backward()
            opt.step()
            losses.update(loss.item())

        # Validation
        with torch.set_grad_enabled(False):
            for local_batch, local_labels in validation_loader:
            
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)

                out= model(local_batch)
                loss = criterion(out , local_labels.float().unsqueeze(1))
                val_losses.update(loss.item())
       
        l = losses.avg  
        train_loss.append(l)
        

        l_ =val_losses.avg
        val_loss.append(l_)
        if disp and epoch+1%10 == 0:
          print("epoch n : ",epoch+1," train: ",l)
          print("epoch n : ",epoch+1," val: ",l_,"\n")
          
    print("epoch n : ",epoch+1," train: ",l)
    print("epoch n : ",epoch+1," val: ",l_,"\n")
    return train_loss , val_loss


def plot_res(list_data,list_labels,title,path,display=True):
  for d,l in zip(list_data,list_labels):
      plt.plot(d,label=l)
  plt.title(title)
  plt.legend(loc="upper left")
  if display:
    plt.show()
  else:
    print('saving')
    plt.savefig(path+'/'+title+'.png')
    plt.clf()


def run (loaders,models,lin,mlp_model,nb_epochs,device,criterion,seed,mlp_use=True,display=True,learning_rate = 0.001):
  torch.manual_seed(seed)
  n_epochs = nb_epochs
  train_info = {}
  for loader, model in zip(loaders,models):
    if model == 'RESNET':
      mlp_model =MLP2(2048,384,1,torch.nn.ReLU)
      lin = Net(2048,1)
    model_infos = {}
    print(f'training linear probing :{model} for {n_epochs} epochs')
    input_shape =loader[0].dataset.__getitem__(2)[0].shape[0]
    #lin = Net(input_shape)
    opt = torch.optim.Adam(lin.parameters(), lr=learning_rate)
    model_infos['linear_train_loss'] , model_infos['linear_val_loss'] = train_loop(lin,n_epochs,criterion,opt,loader[0],loader[1],device,disp=True)
    if mlp_use : 
      print(f"traing {model} MLP ...")
      #mlp = MLP2(input_shape,384,1,torch.nn.ReLU)
      opt = torch.optim.Adam(mlp_model.parameters(), lr=learning_rate)
      model_infos['mlp_train_loss'] , model_infos['mlp_val_loss'] = train_loop(mlp_model,n_epochs,criterion,opt,loader[0],loader[1],device,disp=True)
    train_info[model]=model_infos

  
  return train_info

def train_loop_cls(model,max_epochs,criterion,opt,train_loader,validation_loader,device,disp=False):
    model.to(device)
  
    train_loss , val_loss , train_acc , val_acc = [],[],[],[]
    for epoch in range(max_epochs):
        # Training
        losses = AverageMeter()
        accuracy = AverageMeter()
        val_accuracy = AverageMeter()
        val_losses = AverageMeter()
        for local_batch, local_labels in train_loader:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.type(torch.long).to(device)
            out = model(local_batch)
            
            loss = criterion(out , local_labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.update(loss.item())

            ac = torch.sum(torch.argmax(out,dim=1) == local_labels ).type(torch.float64)/out.shape[0]

            accuracy.update(ac.item())
        # Validation
        with torch.set_grad_enabled(False):
            for local_batch, local_labels in validation_loader:
            
                local_batch, local_labels = local_batch.to(device), local_labels.type(torch.long).to(device)

                out= model(local_batch)
                loss = criterion(out , local_labels)
                val_losses.update(loss.item())

                val_accuracy.update((torch.sum(torch.argmax(out,dim=1) == local_labels ).type(torch.float64)/out.shape[0]).item())
                #print('acc ',(torch.sum(torch.argmax(out,dim=1) == local_labels )/out.shape[0]))

        l = losses.avg  
        train_loss.append(l)
        a=accuracy.avg 
        train_acc.append(a)

        l_ =val_losses.avg
        val_loss.append(l_)
        a_=val_accuracy.avg 
        val_acc.append(a_)
        if disp and epoch%5 == 0:
          print("epoch n : ",epoch+1," train_loss: ",l ,"accuracy : ",a)
          print("epoch n : ",epoch+1," val: ",l_,"accuracy : ",a_,"\n")

    print("epoch n : ",epoch+1," train_loss: ",l ,"accuracy : ",train_acc[-1])
    print("epoch n : ",epoch+1," val: ",l_,"accuracy : ",val_acc[-1],"\n")
    return train_loss , val_loss , train_acc , val_acc


  

def run_cls (loaders,models,nb_epochs,device,seed,out_shape,mlp=False,display=True):
  torch.manual_seed(seed)
  n_epochs = nb_epochs
  
  learning_rate = 0.001
  train_info = {}
  net = None
  for loader, model in zip(loaders,models):
    model_infos = {}
    print(f'training linear probing :{model} for {n_epochs} epochs')
    input_shape =loader[0].dataset.__getitem__(2)[0].shape[0]
    lin = Net(input_shape,out_shape)
    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(lin.parameters(), lr=learning_rate)
    model_infos['linear_train_loss'] , model_infos['linear_val_loss'],model_infos['linear_train_acc'] , model_infos['linear_val_acc'] = train_loop_cls(lin,n_epochs,criterion,opt,loader[0],loader[1],device,disp=False)
    if mlp : 
      print(f'training mlp probing :{model} for {n_epochs} epochs')
      net = MLP2(input_shape,384,out_shape,torch.nn.ReLU)
      opt = torch.optim.Adam(net.parameters(), lr=learning_rate)
      model_infos['mlp_train_loss'] , model_infos['mlp_val_loss'] ,model_infos['mlp_linear_train_acc'] , model_infos['mlp_linear_val_acc']= train_loop_cls(net,n_epochs,criterion,opt,loader[0],loader[1],device,disp=False)
    train_info[model]=model_infos

  return train_info,lin,net


def compare_probe_cls(model1, model2, max_epochs,criterion,opt1, opt2,train_load1, train_load2, val_load1, val_load2,device,disp=False):
    model1.to(device)
    model2.to(device)
    train_loss , val_loss , train_acc , val_acc = [],[],[],[]
    err1 = []
    err2 = []
    for epoch in range(max_epochs):
        # Training
        losses = AverageMeter()
        accuracy = AverageMeter()
        val_accuracy = AverageMeter()
        val_losses = AverageMeter()
        for (local_batch1, local_labels1), (local_batch2, local_labels2) in zip(train_load1, train_load2):
            # Transfer to GPU
            local_batch1, local_batch2, local_labels1 =local_batch1.to(device), local_batch2.to(device), local_labels1.type(torch.long).to(device)
            out1 = model1(local_batch1)
            out2 = model2(local_batch2)
            loss1 = criterion(out1 , local_labels1)
            loss2 = criterion(out2 , local_labels1)
            opt1.zero_grad()
            loss1.backward()
            opt1.step()
            opt2.zero_grad()
            loss2.backward()
            opt2.step()
        
        if epoch == max_epochs-1: 
            print('Final epoch')
        #err1 = []
        #err2 = []
        with torch.set_grad_enabled(False):
            for i, ((local_batch1, local_labels1), (local_batch2, local_labels2)) in enumerate(zip(val_load1, val_load2)):
                assert local_labels1 == local_labels2
                local_batch1, local_batch2, local_labels1 =local_batch1.to(device), local_batch2.to(device), local_labels1.type(torch.long).to(device)
                #print(local_labels1, local_labels2)
                out1= model2(local_batch1)
                loss1 = criterion(out1 , local_labels1)
                out2= model2(local_batch2)
                loss2 = criterion(out2 , local_labels1)
                acc1 = torch.sum(torch.argmax(out1, dim=1) == local_labels1).type(torch.float64)/out1.shape[0]
                acc2 = torch.sum(torch.argmax(out2, dim=1) == local_labels1).type(torch.float64)/out2.shape[0]
                if acc1 != acc2:
                    if int(acc1.item()) ==  local_labels1.item():
                        err2.append(i)
                    if int(acc2.item()) == local_labels1.item():
                        err1.append(i)
        
    print(len(err1), len(err2))
    new_err1 = []
    err1_unique = list(set(err1))
    new_err2 = []
    err2_unique = list(set(err2))
    
    for el in err1_unique:
        if err1.count(el)>12:
            new_err1.append(el)
    for el in err2_unique:
        if err2.count(el)>8:
            new_err2.append(el)
    print('new', len(new_err1), len(new_err2))
    return new_err1, new_err2

def plot_confusion_matrix(cm, classes, file_name, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    #for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #    plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(file_name)


def confusion_cls(model,max_epochs,criterion,opt,train_loader,validation_loader,device,disp=False):
    model.to(device)
    targets = []
    preds = []
    train_loss , val_loss , train_acc , val_acc = [],[],[],[]
    for epoch in range(max_epochs):
        # Training
        losses = AverageMeter()
        accuracy = AverageMeter()
        val_accuracy = AverageMeter()
        val_losses = AverageMeter()
        for local_batch, local_labels in train_loader:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.type(torch.long).to(device)
            out = model(local_batch)

            loss = criterion(out , local_labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.update(loss.item())

            ac = torch.sum(torch.argmax(out,dim=1) == local_labels ).type(torch.float64)/out.shape[0]

            accuracy.update(ac.item())
        # Validation
        with torch.set_grad_enabled(False):
            for local_batch, local_labels in validation_loader:

                local_batch, local_labels = local_batch.to(device), local_labels.type(torch.long).to(device)

                out= model(local_batch)
                loss = criterion(out , local_labels)
                val_losses.update(loss.item())
                
                targets.append(int(local_labels.item()))
                preds.append(int(torch.argmax(out,dim=1).item()))
                val_accuracy.update((torch.sum(torch.argmax(out,dim=1) == local_labels ).type(torch.float64)/out.shape[0]).item())
                #print('acc ',(torch.sum(torch.argmax(out,dim=1) == local_labels )/out.shape[0]))

    stacked = torch.stack((torch.Tensor(targets).int(), torch.Tensor(preds).int()), dim=1)
    cmt = torch.zeros(out.shape[1],out.shape[1], dtype=torch.int64)
    for p in stacked:
        tl, pl = p.tolist()
        #print(tl, pl)
        cmt[tl, pl] = cmt[tl, pl] + 1
    torch.set_printoptions(profile="full")
    print(cmt)
    return cmt
