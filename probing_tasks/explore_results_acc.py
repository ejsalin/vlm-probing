import pickle 
import torch 
import argparse
from utils import *
import sys 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--path",
                        type=str, required=True,
                        help="The input train corpus.")

    parser.add_argument("--output",
                        type=str, required=True,
                        help="The input train corpus.")


    args = parser.parse_args()

    with open(args.path, 'rb') as handle:
        data = pickle.load(handle) 


    means = {model : {'train_lin':[] ,'val_lin':[],'train_mlp':[],'val_mlp':[],'train_lin_acc':[] ,'val_lin_acc':[],'train_mlp_acc':[],'val_mlp_acc':[]} for model in data[list(data.keys())[0]].keys()}
    mean = {model : {'train_lin':0 ,'val_lin':0,'train_mlp':0,'val_mlp':0,'train_lin_acc':0 ,'val_lin_acc':0,'train_mlp_acc':0,'val_mlp_acc':0} for model in data[list(data.keys())[0]].keys()}
    means_last = {model : {'train_lin':[] ,'val_lin':[],'train_mlp':[],'val_mlp':[],'train_lin_acc':[] ,'val_lin_acc':[],'train_mlp_acc':[],'val_mlp_acc':[]} for model in data[list(data.keys())[0]].keys()}
    mean_last = {model : {'train_lin':0 ,'val_lin':0,'train_mlp':0,'val_mlp':0,'train_lin_acc':0 ,'val_lin_acc':0,'train_mlp_acc':0,'val_mlp_acc':0} for model in data[list(data.keys())[0]].keys()}
    means_idx = {model : {'train_lin':[] ,'val_lin':[],'train_mlp':[],'val_mlp':[],'train_lin_acc':[] ,'val_lin_acc':[],'train_mlp_acc':[],'val_mlp_acc':[]} for model in data[list(data.keys())[0]].keys()}
    mean_idx = {model : {'train_lin':0 ,'val_lin':0,'train_mlp':0,'val_mlp':0,'train_lin_acc':0 ,'val_lin_acc':0,'train_mlp_acc':0,'val_mlp_acc':0} for model in data[list(data.keys())[0]].keys()}

    for run in data.keys():
      if isinstance(run,int): 
        for model in data[run].keys():

            means[model]['train_lin'].append(min(data[run][model]['linear_train_loss']))
            means[model]['val_lin'].append(min(data[run][model]['linear_val_loss']))

            means[model]['train_mlp'].append(min(data[run][model]['mlp_train_loss']))
            means[model]['val_mlp'].append(min(data[run][model]['mlp_val_loss']))

            means[model]['train_lin_acc'].append(max(data[run][model]['linear_train_acc'])*100)
            means[model]['val_lin_acc'].append(max(data[run][model]['linear_val_acc'])*100)
            
            means[model]['train_mlp_acc'].append(max(data[run][model]['mlp_linear_train_acc'])*100)
            means[model]['val_mlp_acc'].append(max(data[run][model]['mlp_linear_val_acc'])*100)

            means_last[model]['train_mlp_acc'].append(data[run][model]['mlp_linear_train_acc'][-1]*100)
            means_last[model]['val_mlp_acc'].append(data[run][model]['mlp_linear_val_acc'][-1]*100)

            means_last[model]['train_lin'].append(data[run][model]['linear_train_loss'][-1])
            means_last[model]['val_lin'].append(data[run][model]['linear_val_loss'][-1])

            means_last[model]['train_mlp'].append(data[run][model]['mlp_train_loss'][-1])
            means_last[model]['val_mlp'].append(data[run][model]['mlp_val_loss'][-1])

            means_last[model]['train_lin_acc'].append(data[run][model]['linear_train_acc'][-1]*100)
            means_last[model]['val_lin_acc'].append(data[run][model]['linear_val_acc'][-1]*100)

            means_idx[model]['train_lin'].append(data[run][model]['linear_train_loss'].index(min(data[run][model]['linear_train_loss'])))
            means_idx[model]['val_lin'].append(data[run][model]['linear_val_loss'].index(min(data[run][model]['linear_val_loss'])))

            means_idx[model]['train_mlp'].append(data[run][model]['mlp_train_loss'].index(min(data[run][model]['mlp_train_loss'])))
            means_idx[model]['val_mlp'].append(data[run][model]['mlp_val_loss'].index(min(data[run][model]['mlp_val_loss'])))

            means_idx[model]['train_lin_acc'].append(data[run][model]['linear_train_acc'].index(max(data[run][model]['linear_train_acc'])))
            means_idx[model]['val_lin_acc'].append(data[run][model]['linear_val_acc'].index(max(data[run][model]['linear_val_acc'])))
            
            means_idx[model]['train_mlp_acc'].append(data[run][model]['mlp_linear_train_acc'].index(max(data[run][model]['mlp_linear_train_acc'])))
            means_idx[model]['val_mlp_acc'].append(data[run][model]['mlp_linear_val_acc'].index(max(data[run][model]['mlp_linear_val_acc'])))







    for model in data[list(data.keys())[0]].keys():

        mean[model]['train_lin'] = str(format(np.mean(np.array(means[model]['train_lin'])),'.2f')) +' +- '+ str(format(np.std(np.array(means[model]['train_lin'])),'.2f'))
        mean[model]['val_lin'] =   str(format(np.mean(np.array(means[model]['val_lin'])),'.2f')) +' +- '+ str(format(np.std(np.array(means[model]['val_lin'])),'.2f'))

        mean[model]['train_mlp'] =  str(format(np.mean(np.array(means[model]['train_mlp'])),'.2f')) +' +- '+ str(format(np.std(np.array(means[model]['train_mlp'])),'.2f'))
        mean[model]['val_mlp'] = str(format(np.mean(np.array(means[model]['val_mlp'])),'.2f')) +' +- '+ str(format(np.std(np.array(means[model]['val_mlp'])),'.2f'))

        mean[model]['train_lin_acc'] = str(format(np.mean(np.array(means[model]['train_lin_acc'])),'.2f')) +' +- '+ str(format(np.std(np.array(means[model]['train_lin'])),'.2f'))
        mean[model]['val_lin_acc'] =   str(format(np.mean(np.array(means[model]['val_lin_acc'])),'.2f')) +' +- '+ str(format(np.std(np.array(means[model]['val_lin'])),'.2f'))

        mean[model]['train_mlp_acc'] =  str(format(np.mean(np.array(means[model]['train_mlp_acc'])),'.2f')) +' +- '+ str(format(np.std(np.array(means[model]['train_mlp_acc'])),'.2f'))
        mean[model]['val_mlp_acc'] = str(format(np.mean(np.array(means[model]['val_mlp_acc'])),'.2f')) +' +- '+ str(format(np.std(np.array(means[model]['val_mlp_acc'])),'.2f'))


        mean_last[model]['train_lin'] = str(format(np.mean(np.array(means_last[model]['train_lin'])),'.2f')) +' +- '+ str(format(np.std(np.array(means_last[model]['train_lin'])),'.2f'))
        mean_last[model]['val_lin'] =   str(format(np.mean(np.array(means_last[model]['val_lin'])),'.2f')) +' +- '+ str(format(np.std(np.array(means_last[model]['val_lin'])),'.2f'))

        mean_last[model]['train_mlp'] =  str(format(np.mean(np.array(means_last[model]['train_mlp'])),'.2f')) +' +- '+ str(format(np.std(np.array(means_last[model]['train_mlp'])),'.2f'))
        mean_last[model]['val_mlp'] = str(format(np.mean(np.array(means_last[model]['val_mlp'])),'.2f')) +' +- '+ str(format(np.std(np.array(means_last[model]['val_mlp'])),'.2f'))

        mean_last[model]['train_lin_acc'] = str(format(np.mean(np.array(means_last[model]['train_lin_acc'])),'.2f')) +' +- '+ str(format(np.std(np.array(means_last[model]['train_lin'])),'.2f'))
        mean_last[model]['val_lin_acc'] =   str(format(np.mean(np.array(means_last[model]['val_lin_acc'])),'.2f')) +' +- '+ str(format(np.std(np.array(means_last[model]['val_lin'])),'.2f'))

        mean_last[model]['train_mlp_acc'] =  str(format(np.mean(np.array(means_last[model]['train_mlp_acc'])),'.2f')) +' +- '+ str(format(np.std(np.array(means_last[model]['train_mlp_acc'])),'.2f'))
        mean_last[model]['val_mlp_acc'] = str(format(np.mean(np.array(means_last[model]['val_mlp_acc'])),'.2f')) +' +- '+ str(format(np.std(np.array(means_last[model]['val_mlp_acc'])),'.2f'))


        mean_idx[model]['train_lin'] = str(format(np.mean(np.array(means_idx[model]['train_lin'])),'.1f'))
        mean_idx[model]['val_lin'] =   str(format(np.mean(np.array(means_idx[model]['val_lin'])),'.1f'))

        mean_idx[model]['train_mlp'] =  str(format(np.mean(np.array(means_idx[model]['train_mlp'])),'.1f'))
        mean_idx[model]['val_mlp'] = str(format(np.mean(np.array(means_idx[model]['val_mlp'])),'.1f'))

        mean_idx[model]['train_lin_acc'] = str(format(np.mean(np.array(means_idx[model]['train_lin_acc'])),'.1f'))
        mean_idx[model]['val_lin_acc'] =   str(format(np.mean(np.array(means_idx[model]['val_lin_acc'])),'.1f'))

        mean_idx[model]['train_mlp_acc'] =  str(format(np.mean(np.array(means_idx[model]['train_mlp_acc'])),'.1f'))
        mean_idx[model]['val_mlp_acc'] = str(format(np.mean(np.array(means_idx[model]['val_mlp_acc'])),'.1f'))



    
    import os

    # define the name of the directory to be created
    output = args.output
    original_stdout = sys.stdout
    try:
        os.makedirs(output)
    except OSError:
        print ("Creation of the directory %s failed" % output)
    else:
        print ("Successfully created the directory %s" % output)

    with open(args.output+'/results.txt', 'w') as f:
        sys.stdout = f 

        for model in means.keys():
            print(model,'_________________________________')
            print('train#')
            print('lin-loss : ',mean[model]['train_lin']+" last :"+mean_last[model]['train_lin']+"  epoch : "+mean_idx[model]['train_lin'])
            #print('mlp : ',mean[model]['train_mlp']+" last :"+mean_last[model]['train_mlp']+"  epoch : "+mean_idx[model]['train_mlp'])
            print('lin-acc : ',mean[model]['train_lin_acc']+" last :"+mean_last[model]['train_lin_acc']+"  epoch : "+mean_idx[model]['train_lin_acc'])
            #print('mlp : ',mean[model]['train_mlp_acc']+" last :"+mean_last[model]['train_mlp_acc']+"  epoch : "+mean_idx[model]['train_mlp_acc'])
            print()
            print('val#')
            print('lin-loss : ',mean[model]['val_lin']+" last :"+mean_last[model]['val_lin']+"  epoch : "+mean_idx[model]['val_lin'])
            #print('mlp : ',mean[model]['val_mlp']+" last :"+mean_last[model]['val_mlp']+"  epoch : "+mean_idx[model]['val_mlp'])
            print('lin-acc : ',mean[model]['val_lin_acc']+" last :"+mean_last[model]['val_lin_acc']+"  epoch : "+mean_idx[model]['val_lin_acc'])
            #print('mlp : ',mean[model]['val_mlp_acc']+" last :"+mean_last[model]['val_mlp_acc']+"  epoch : "+mean_idx[model]['val_mlp_acc'])
            print()

    # plot losses
        sys.stdout = original_stdout


    
    #for model in data[list(data.keys())[0]].keys() :
    #    m = data[list(data.keys())[0]][model]
        #plot_res([m['linear_train_loss'],m['linear_val_loss']],['train','val'],f"{model} linear loss",output,display=False)
        #plot_res([m['mlp_train_loss'],m['mlp_val_loss']],['train','val'],f"{model} mlp loss",output,display=False)


    #plot_res([data[list(data.keys())[0]][model]['linear_val_loss'] for model in data[list(data.keys())[0]].keys() ],
    #        [model for model in data[list(data.keys())[0]].keys()],"models val linear loss",output,display=False )
#
    #plot_res([data[list(data.keys())[0]][model]['mlp_val_loss'] for model in data[list(data.keys())[0]].keys() ],
    #        [model for model in data[list(data.keys())[0]].keys()],"models val mlp loss",output,display=False )
#
    #plot_res([data[list(data.keys())[0]][model]['linear_val_acc'] for model in data[list(data.keys())[0]].keys() ],
    #        [model for model in data[list(data.keys())[0]].keys()],"models val linear accuracy",output,display=False )
#
    #plot_res([data[list(data.keys())[0]][model]['mlp_linear_val_acc'] for model in data[list(data.keys())[0]].keys() ],
    #        [model for model in data[list(data.keys())[0]].keys()],"models val mlp accuracy",output,display=False )