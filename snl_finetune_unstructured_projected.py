# Selective Network Linearization unstructured method.
# Starting from the pretrained model. 

import argparse
import os
from datasets import get_dataset, DATASETS
from architectures_unstructured import ARCHITECTURES, get_architecture
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
import copy
from tqdm.rich import tqdm
from train_utils import init_logfile, log
from utils import *
import sys

from snip import SNIP
from archs_unstructured.cifar_resnet import LearnableAlpha

from accelerate import Accelerator
from accelerate.utils import set_seed
set_seed(10)
accelerator = Accelerator()



parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument('arch', type=str, choices=ARCHITECTURES)
parser.add_argument('outdir', type=str, help='folder to save model and training log)')
parser.add_argument('savedir', type=str, help='folder to load model')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--finetune_epochs', default=100, type=int,
                    help='number of total epochs for the finetuning')
parser.add_argument('--epochs', default=300, type=int)
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
parser.add_argument('--type', type=str, default='senet', choices=['senet', 'privshap'], 
                    help='Type of baseline (senet or privshap)')
args = parser.parse_args()

if args.budegt_type == 'relative' and args.relu_budget > 1:
    print(f'Warning: relative budget type is used, but the relu budget is {args.relu_budget} > 1.')
    sys.exit(1)

def _collect_alpha_params(module: nn.Module) -> list[torch.Tensor]:
    """Collect all alpha parameters from LearnableAlpha layers in deterministic order."""
    alphas = []
    for m in module.modules():
        if isinstance(m, LearnableAlpha):
            alphas.append(m.alphas)
    return alphas

def relu_counting(net, args):
    """Count the number of active ReLUs based on alpha threshold."""
    relu_count = 0
    for name, param in net.named_parameters():
        if 'alpha' in name:
            boolean_list = param.data > args.threshold
            relu_count += (boolean_list == 1).sum()
    return relu_count

def project_space_wrn(net, budget_list):
    alphas = _collect_alpha_params(net)
    if len(budget_list) != len(alphas):
        raise ValueError(f"Budget list length {len(budget_list)} != number of alpha modules {len(alphas)}")
    for alpha, budget in zip(alphas, budget_list):
        if alpha.grad is None:
            alpha.data.zero_()
            continue
        grad = torch.flatten(torch.abs(alpha.grad))
        budget = min(budget, len(grad))
        if budget == 0:
            alpha.data.zero_()
            continue
        thres_tensor, _ = torch.topk(grad, budget, sorted=True)
        thres = thres_tensor[-1]
        alpha.data = (torch.abs(alpha.grad) > thres).float()
def project_space(net, budget_list):
    alphas = _collect_alpha_params(net)
    if len(budget_list) != len(alphas):
        raise ValueError(f"Budget list length {len(budget_list)} != number of alpha modules {len(alphas)}")
    for alpha, budget in zip(alphas, budget_list):
        if alpha.grad is None:
            alpha.data.zero_()
            continue
        grad = torch.flatten(torch.abs(alpha.grad))
        budget = min(budget, len(grad))
        if budget == 0:
            alpha.data.zero_()
            continue
        thres_tensor, _ = torch.topk(grad, budget, sorted=True)
        thres = thres_tensor[-1]
        alpha.data = (torch.abs(alpha.grad) > thres).float()

def build_dataloaders(dataset_name: str, batch_size: int, workers: int) -> tuple[DataLoader, DataLoader]:
    """Build train and test dataloaders for the given dataset."""
    train_dataset = get_dataset(dataset_name, 'train')
    test_dataset = get_dataset(dataset_name, 'test')
    pin_memory = (dataset_name == "imagenet")
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
                              num_workers=workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size,
                             num_workers=workers, pin_memory=pin_memory)
    return train_loader, test_loader

def load_base_model(arch: str, dataset: str, device: torch.device, args) -> nn.Module:
    """Load the pretrained base model from checkpoint."""
    base_classifier = get_architecture(arch, dataset, device, args)
    if not os.path.isfile(args.savedir):
        raise FileNotFoundError(f"Checkpoint not found at {args.savedir}")
    checkpoint = torch.load(args.savedir, map_location=device)
    base_classifier.load_state_dict(checkpoint['state_dict'])
    base_classifier.eval()
    return base_classifier
    
def main():
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
    log(logfilename, "SNIP Type: {:}".format(args.snip_type))

    train_loader, test_loader = build_dataloaders(args.dataset, args.batch, args.workers)
    base_classifier = load_base_model(args.arch, args.dataset, device, args)
    net = copy.deepcopy(base_classifier)



    log(logfilename, "Loaded the base_classifier")

    # Calculating the loaded model's test accuracy.
    original_acc = model_inference(base_classifier, test_loader,
                                    device, display=True)
    
    log(logfilename, "Original Model Test Accuracy: {:.5}".format(original_acc))

    # Creating a fresh copy of network not affecting the original network.

    net = net.to(device)

    relu_count = relu_counting(net, args)

    log(logfilename, "Original ReLU Count: {}".format(relu_count))
    budgets_list = SNIP(net, args.relu_budget/relu_count, train_loader, device, args.type)
        # reludic[budget] = budgets_list
    print('relu budgets : ',budgets_list)
    print("total relu budges: ", sum(budgets_list))

    # Alpha is the masking parameters initialized to 1. Enabling the grad.
    for name, param in net.named_parameters():
        # param.requires_grad = False
        if 'alpha' in name:
            param.requires_grad = True
        
    criterion = nn.CrossEntropyLoss().to(device)  
    optimizer = Adam(net.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[args.epochs // 2,  3*args.epochs // 4], last_epoch=-1)
    # scheduler = CosineAnnealingLR(optimizer, T_max = 50)
    # scheduler = StepLR(optimizer, step_size = 30, gamma=0.1)
    
    # counting number of ReLU.
    total = relu_counting(net, args)
    if args.budegt_type == 'relative':
        args.relu_budget = int(total * args.relu_budget)

    lowest_relu_count, relu_count = total, total
    for epoch in tqdm(range(args.epochs)):


        train_loss = mask_train_kd_unstructured(train_loader, net, base_classifier, criterion, optimizer, 
                                epoch, device, alpha=args.alpha, display=False)

        scheduler.step()
        project_space_wrn(net, budgets_list)
        acc = model_inference(net, test_loader, device, display=False)
        relu_count = relu_counting(net, args)        
        log(logfilename, 'Epochs: {}\t'
              'Test Acc: {}\t'
              'Relu Count: {}\t'
              'Alpha: {:.6f}\t'.format(
                  epoch, acc, relu_count, args.alpha
              )
              )
        
        # if relu_count < lowest_relu_count:
        #     lowest_relu_count = relu_count 
        
        # elif relu_count >= lowest_relu_count and epoch >= 5:
        #     args.alpha *= 1.1

        # if relu_count <= args.relu_budget:
        #     print("Current epochs breaking loop at {:}".format(epoch))
        #     break


    for name, param in net.named_parameters():
        if 'alpha' in name:
            # boolean_list = param.data > args.threshold
            # param.data = boolean_list.float()
            param.requires_grad = False

 
    # finetune_epoch = args.finetune_epochs

    # optimizer = SGD(net.parameters(), lr=1e-3, momentum=args.momentum, weight_decay=args.weight_decay)
    # criterion = nn.CrossEntropyLoss().to(device)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, finetune_epoch)
    
    # print("Finetuning the model")
    # log(logfilename, "Finetuning the model")

    # best_top1 = 0
    # for epoch in tqdm(range(finetune_epoch)):
    #     train_loss, train_top1, train_top5 = train_kd(train_loader, net, base_classifier, optimizer, criterion, epoch, device)
    #     test_loss, test_top1, test_top5 = test(test_loader, net, criterion, device, 100, display=True)
    #     scheduler.step()
        
    #     if best_top1 < test_top1:
    #         best_top1 = test_top1
    #         is_best = True
    #     else:
    #         is_best = False

    #     if is_best:
    #         torch.save({
    #                 'arch': args.arch,
    #                 'state_dict': net.state_dict(),
    #                 'optimizer': optimizer.state_dict(),
    #         }, os.path.join(args.outdir, f'snl_best_checkpoint_{args.arch}_{args.dataset}_{args.relu_budget}.pth.tar'))

    # print("Final best Prec@1 = {}%".format(best_top1))
    # log(logfilename, "Final best Prec@1 = {}%".format(best_top1))
    
        
if __name__ == "__main__":
    main()
