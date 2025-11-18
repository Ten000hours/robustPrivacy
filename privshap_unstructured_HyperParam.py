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
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import datetime
import time as pytime
import numpy as np
import copy
from tqdm.rich import tqdm
from train_utils import AverageMeter, accuracy, accuracy_list, init_logfile, log
from utils import *
import sys
import optuna

from optuna.trial import TrialState
from snip import SNIP
from archs_unstructured.cifar_resnet import BasicBlock, BasicBlock_IN, LearnableAlpha


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

# Deprecated global index used for projection ordering. Replaced by
# deterministic module collection in project_space().

if args.budegt_type == 'relative' and args.relu_budget > 1:
    print(f'Warning: relative budget type is used, but the relu budget is {args.relu_budget} > 1.')
    sys.exit(1)

def relu_counting(net, args):
    """Count active ReLUs based on current alpha thresholds."""
    relu_count = 0
    for name, param in net.named_parameters():
        if 'alpha' in name:
            boolean_list = param.data > args.threshold
            relu_count += (boolean_list == 1).sum()
    return relu_count

def build_dataloaders(dataset: str, batch_size: int, workers: int):
    """Create train and test dataloaders for the given dataset.

    Returns train_loader and test_loader with appropriate pin_memory for imagenet.
    """
    train_dataset = get_dataset(dataset, 'train')
    test_dataset = get_dataset(dataset, 'test')
    pin_memory = (dataset == "imagenet")
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
                              num_workers=workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size,
                             num_workers=workers, pin_memory=pin_memory)
    return train_loader, test_loader

def load_base_model(arch: str, dataset: str, device: torch.device, args):
    """Load architecture and weights from args.savedir as eval-ready model.

    Raises FileNotFoundError if checkpoint path does not exist.
    """
    if not os.path.exists(args.savedir):
        raise FileNotFoundError(f"Checkpoint not found at {args.savedir}")
    base_classifier = get_architecture(arch, dataset, device, args)
    checkpoint = torch.load(args.savedir, map_location=device)
    base_classifier.load_state_dict(checkpoint['state_dict'])
    base_classifier.eval()
    return base_classifier

def _collect_alpha_modules(module):
    """Collect LearnableAlpha modules in a stable, recursive order.

    """
    modules = []
    for name, child in module.named_children():
        if isinstance(child, (nn.Sequential, BasicBlock_IN)):
            modules.extend(_collect_alpha_modules(child))
        elif isinstance(child, LearnableAlpha):
            modules.append(child)
        else:
            # Recurse into nested containers if present
            if len(list(child.named_children())) > 0:
                modules.extend(_collect_alpha_modules(child))
    return modules

def project_space(net, budget_list):
    """Project the gradient magnitudes of LearnableAlpha to a binary mask.

    For each LearnableAlpha module, select the top-k entries (by absolute
    gradient) defined by the corresponding item in budget_list and set the
    alpha mask to 1 for those entries, 0 otherwise. 
    """
    alpha_modules = _collect_alpha_modules(net)
    if len(alpha_modules) == 0:
        return

    # Validate length and adjust budgets if necessary
    if len(budget_list) != len(alpha_modules):
        print(f"[project_space] Warning: budgets ({len(budget_list)}) != alpha modules ({len(alpha_modules)}).")
        if len(budget_list) < len(alpha_modules):
            last = budget_list[-1] if len(budget_list) > 0 else 0
            pad = [last] * (len(alpha_modules) - len(budget_list))
            budget_list = list(budget_list) + pad
        else:
            budget_list = list(budget_list)[:len(alpha_modules)]

    for layer, budget in zip(alpha_modules, budget_list):
        abs_values = torch.flatten(torch.abs(layer.alphas.grad))
        if abs_values.numel() == 0:
            # No gradient information; default to zeros
            layer.alphas.data = torch.zeros_like(layer.alphas)
            continue

        # Guard against invalid budgets
        budget = int(max(0, min(budget, abs_values.numel())))
        if budget == 0:
            layer.alphas.data = torch.zeros_like(layer.alphas)
            continue

        thres_tensor, _ = torch.topk(abs_values, budget, sorted=True)
        thres = thres_tensor[-1]
        layer.alphas.data = (torch.abs(layer.alphas.grad) > thres).float()
    
def objective(trial: optuna.trial.Trial):
    """Optuna objective that runs SNL finetuning with unstructured masking.

    Tunes optimizer choice and learning rate, projects LearnableAlpha masks
    using SNIP budgets, and reports test accuracy to Optuna.
    """
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

    train_loader, test_loader = build_dataloaders(args.dataset, args.batch, args.workers)


    # Loading the base_classifier
    base_classifier = load_base_model(args.arch, args.dataset, device, args)
    net = copy.deepcopy(base_classifier)


    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "AdamW"])
    lr = trial.suggest_categorical("lr", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
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
    # Use the optimizer configured by Optuna above
    # scheduler can optionally be enabled using args.lr_step_size and args.gamma
    # scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)
    
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
    
        
# if __name__ == "__main__":
    
#     study = optuna.create_study(direction="maximize")
#     study.optimize(objective, n_trials=30)

#     pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
#     complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

#     print("Study statistics: ")
#     print("  Number of finished trials: ", len(study.trials))
#     print("  Number of pruned trials: ", len(pruned_trials))
#     print("  Number of complete trials: ", len(complete_trials))

#     print("Best trial:")
#     trial = study.best_trial

#     print("  Value: ", trial.value)

#     print("  Params: ")
#     for key, value in trial.params.items():
#         print("    {}: {}".format(key, value))