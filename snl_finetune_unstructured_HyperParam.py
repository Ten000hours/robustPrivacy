# Selective Network Linearization unstructured method.
# Starting from the pretrained model. 

import argparse
import os
from datasets import get_dataset, DATASETS, get_num_classes
from architectures_unstructured import ARCHITECTURES, get_architecture
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD, Optimizer, Adam,AdamW
from torch.optim.lr_scheduler import StepLR
import datetime
import time
import numpy as np
import copy
import types
from tqdm.rich import tqdm
import time
from math import ceil
from train_utils import AverageMeter, accuracy, accuracy_list, init_logfile, log
from utils import *
import sys

import optuna
from optuna.trial import TrialState
import torch.optim as optim
import neptune
from snip import SNIP
from archs_unstructured.cifar_resnet import BasicBlock, BasicBlock_IN, LearnableAlpha



# mlflow.autolog()
# run = neptune.init_run(
#     project="xiangruixu1/snip-snl-pgd",
#     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiOGQ0M2JmYi1jNDMwLTQyZjItOTI1Ni1iYTI3NzFmZDQ2NjIifQ==",
# )  # your credentials


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument('arch', type=str, choices=ARCHITECTURES)
parser.add_argument('outdir', type=str, help='folder to save model and training log)')
parser.add_argument('savedir', type=str, help='folder to load model')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--finetune_epochs', default=100, type=int,
                    help='number of total epochs for the finetuning')
parser.add_argument('--epochs', default=130, type=int)
parser.add_argument('--batch', default=256, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--logname', type=str, default='log.txt')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--alpha', default=1e-5, type=float,
                    help='Lasso coefficient')
parser.add_argument('--threshold', default=1e-2, type=float)
parser.add_argument('--budegt_type', default='absolute', type=str, choices=['absolute', 'relative'])
parser.add_argument('--relu_budget', default=50000, type=int)
parser.add_argument('--lr_step_size', type=int, default=30,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
# parser.add_argument('--gpus', default=1, type=int,
#                     help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--stride', type=int, default=1, help='conv1 stride')
args = parser.parse_args()

index = 0

if args.budegt_type == 'relative' and args.relu_budget > 1:
    print(f'Warning: relative budget type is used, but the relu budget is {args.relu_budget} > 1.')
    sys.exit(1)

def relu_counting(net, args):
    relu_count = 0
    for name, param in net.named_parameters():
        if 'alpha' in name:
            boolean_list = param.data > args.threshold
            relu_count += (boolean_list == 1).sum()
    return relu_count

def project_space(net, budget_list):
    # for name, param in net.named_parameters():
    #     if 'alpha' in name:
    #         abs_values = torch.abs(param.data)
    #         param.data = torch.topk(abs_values, budget_list[index], sorted=False)
    #     index += 1
    for name, layer in net.named_children():
        if isinstance(layer, nn.Sequential) or isinstance(layer, BasicBlock_IN):
            project_space(layer, budget_list)
        elif isinstance(layer, LearnableAlpha):
            # print('name: ',name, 'and layer:',layer)
            abs_values = torch.flatten(torch.abs(layer.alphas.grad))
            # print('abs values: ',abs_values)
            global index
            # print('abs values: ',len(abs_values),'and budget ', budget_list[index])
            thres_tensor,_ = torch.topk(abs_values, budget_list[index], sorted=True)
            thres = thres_tensor[-1]
            layer.alphas.data = (torch.abs(layer.alphas.grad) > thres).float() #* layer.alphas
            index += 1
    
def objective(trial):
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    device = torch.device("cuda")
    # torch.cuda.set_device(args.gpu)


    logfilename = os.path.join(args.outdir, args.logname)

    log(logfilename, "Hyperparameter List")
    log(logfilename, "Finetune Epochs: {:}".format(args.finetune_epochs))
    log(logfilename, "Learning Rate: {:}".format(args.lr))
    log(logfilename, "Alpha: {:}".format(args.alpha))
    log(logfilename, "ReLU Budget: {:}".format(args.relu_budget))

    train_dataset = get_dataset(args.dataset, 'train')
    test_dataset = get_dataset(args.dataset, 'test')
    pin_memory = (args.dataset == "imagenet")
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
                                num_workers=args.workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                                num_workers=args.workers, pin_memory=pin_memory)


    # Loading the base_classifier
    base_classifier = get_architecture(args.arch, args.dataset, device, args)
    net = copy.deepcopy(base_classifier)
    checkpoint = torch.load(args.savedir, map_location=device)
    base_classifier.load_state_dict(checkpoint['state_dict'])
    base_classifier.eval()



    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "AdamW"])
    lr = trial.suggest_categorical("lr",[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    optimizer = getattr(optim, optimizer_name)(net.parameters(), lr=lr)



    log(logfilename, "Loaded the base_classifier")

    # Calculating the loaded model's test accuracy.
    original_acc = model_inference(base_classifier, test_loader,
                                    device, display=True)
    
    log(logfilename, "Original Model Test Accuracy: {:.5}".format(original_acc))

    # Creating a fresh copy of network not affecting the original network.

    net = net.to(device)

    relu_count = relu_counting(net, args)

    log(logfilename, "Original ReLU Count: {}".format(relu_count))
    reluBudgts = [ind for ind in range(args.relu_budget,300000, 50000)]
    reludic = {}
    # budgets_list  is the num of relu to keep
    for budget in reluBudgts: 
        budgets_list = SNIP(net, budget/relu_count, train_loader, device)
        reludic[budget] = budgets_list
    # print('relu budgets : ',budgets_list)

    # Alpha is the masking parameters initialized to 1. Enabling the grad.
    for name, param in net.named_parameters():
        # param.requires_grad = False
        if 'alpha' in name:
            param.requires_grad = True
        
    criterion = nn.CrossEntropyLoss().to(device)  
    optimizer = Adam(net.parameters(), lr=args.lr)
    # scheduler = StepLR(optimizer, step_size = 30, gamma=0.1)
    
    # counting number of ReLU.
    total = relu_counting(net, args)
    if args.budegt_type == 'relative':
        args.relu_budget = int(total * args.relu_budget)

    # Corresponds to Line 4-9
    lowest_relu_count, relu_count = total, total



    for epoch in tqdm(range(args.epochs)):

        # Omask, mask = [],[]
        # for name, param in net.named_parameters():
        #     # param.requires_grad = False
        #     if 'alpha' in name:
        #         Omask.append(param)
        # Omask = torch.cat([w.flatten() for w in Omask])
        # Simultaneous tarining of w and alpha with KD loss.
        train_loss = mask_train_kd_unstructured(train_loader, net, base_classifier, criterion, optimizer, 
                                epoch, device, alpha=args.alpha, display=False)

        # iteratively project each relu layers with budgets_list

        if epoch ==0:
            # budgets_list = reludic[295000]
            budgets_list = reludic[args.relu_budget]
        project_space(net, budgets_list)

        # for name, param in net.named_parameters():
        #     # param.requires_grad = False
        #     if 'alpha' in name:
        #         mask.append(param)
        # mask = torch.cat([w.flatten() for w in mask])
        global index 
        index = 0 

        acc = model_inference(net, test_loader, device, display=False)



        # if 0 <= acc < 45:
        #     budgets_list = reludic[295000]
        # elif 45 <= acc < 50:
        #     budgets_list = reludic[245000]
        # elif 50 <= acc < 55:
        #     budgets_list = reludic[195000]
        # elif 55 <= acc < 60:
        #     budgets_list = reludic[145000]
        # elif 60 <= acc < 64:
        #     budgets_list = reludic[95000]
        # elif acc >=64:
        #     budgets_list = reludic[45000]

        # run["train/accuracy"].append(acc)
        # run["metric"].append(
        #     value=acc,
        #     step=epoch,
        # )
        # counting ReLU in the neural network by using threshold.
        relu_count = relu_counting(net, args)        
        log(logfilename, 'Epochs: {}\t'
              'Test Acc: {}\t'
              'Relu Count: {}\t'
              'Alpha: {:.6f}\t'.format(
                  epoch, acc, relu_count, args.alpha
              )
              )
        trial.report(acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # log(logfilename, "After SNL Algorithm, the current ReLU Count: {}, rel. count:{}".format(relu_count, relu_count/total))


    # Handle pruning based on the intermediate value.

    return acc
    
        
if __name__ == "__main__":
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))