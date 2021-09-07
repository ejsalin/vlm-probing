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
torch.manual_seed(0)


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


  train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                             sampler=train_sampler)

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