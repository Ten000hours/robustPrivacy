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
import itertools
from archs_unstructured.cifar_resnet import BasicBlock, BasicBlock_IN, LearnableAlpha
from rich.console import Console

console = Console()


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument('arch', type=str, choices=ARCHITECTURES)
parser.add_argument('outdir', type=str, help='folder to save model and training log)')
parser.add_argument('savedir', type=str, help='folder to load model')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--batch', default=256, type=int, metavar='N',
                    help='batchsize (default: 256)')
# parser.add_argument('--gpus', default=1, type=int,
#                     help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--logname', type=str, default='log.txt')
# parser.add_argument('--print-freq', default=100, type=int,
#                     metavar='N', help='print frequency (default: 10)')
parser.add_argument('--stride', type=int, default=1, help='conv1 stride')
args = parser.parse_args()

layer_mask = 0


def find_subsets(nums):
    subsets = []
    for i in range(len(nums) + 1):
        subsets.extend(itertools.combinations(nums, i))
    return [list(subset) for subset in subsets]



def mask_model(model, test_loader, device, mask):
    for name, layer in model.named_children():
        if isinstance(layer, nn.Sequential) or isinstance(layer, BasicBlock_IN):
            mask_model(layer, test_loader, device, mask)
        elif isinstance(layer, LearnableAlpha):
            # print('name: ',name, 'and layer:',layer)
            global layer_mask
            if layer_mask not in mask:
                layer.alphas.data = torch.zeros_like(layer.alphas)
            layer_mask += 1
    # return model

def get_utility( test_loader, device, mask, base_classifier):
    # print("==========================================")
    # tempac = model_inference(model, test_loader, device, display=True)

    # print("acc before mask: ", tempac)
    # print("==========================================")
    model = copy.deepcopy(base_classifier)
    model = model.to(device)
    # remove the relu layers based on mask 
    global layer_mask 
    layer_mask = 0
    mask_model(model, test_loader, device, mask)

    return model_inference(model, test_loader, device, display=True)

from scipy.special import comb, perm
# #计算排列数
# A=perm(3,2)
# #计算组合数
# C=comb(45,2)
# print(A,C)

def ith_shapley_value(relu_layers_No, layer_index, model, test_loader, device, base_model):
    subsets = find_subsets([i for i in range(relu_layers_No) if i != layer_index])
    sv =0
    for mask in subsets[::-1]:
        U_s = get_utility( test_loader, device, mask, base_model)
        all_mask = mask +[layer_index]
        U_si = get_utility( test_loader, device, all_mask, base_model)
        sv += (U_si - U_s)/comb(relu_layers_No-1,len(mask))
    sv = sv * 1/relu_layers_No

    return sv

# nums = [1, 2,4,5]
# all_subsets = find_subsets(nums)
# print(all_subsets)


def main():
    logfilename = os.path.join(args.outdir, args.logname)
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    device = torch.device("cuda")
    # torch.cuda.set_device(args.gpu)

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

# shapley values (perm): [0.0566337425595238, 0.09678317212301589, 0.14673694816468255, -0.01661306423611111, -0.0316541728670635, 0.012727182539682545, 0.09678673735119049, -0.02139081101190476]

# shapley value (comb):[10.981988095238092, 11.128154761904765, 11.550011904761906, 11.034011904761893, 10.8500119047619, 10.87782142857142, 6.478488095238097, 0.13951190476190362]

# =====================================================test code============================================================
    # global layer_mask 
    # layer_mask = 0
    # mask = [0,1,2,3,4,5,6,7]
    # print(mask)
    # net = copy.deepcopy(base_classifier)
    # mask_model(net, test_loader, device, mask)
    # acc=model_inference(net, test_loader, device, display=True)
    # print(acc)

    # # log(logfilename, "Loaded the base_classifier")

    # # Calculating the loaded model's test accuracy.
    # original_acc = model_inference(base_classifier, test_loader,
    #                                 device, display=True)
    # print(original_acc)

    
    # ======================================================================================================
    
    # log(logfilename, "Original Model Test Accuracy: {:.5}".format(original_acc))

    # Creating a fresh copy of network not affecting the original network.

    net = net.to(device)
    # print(net)

    res = []
    for i in tqdm(range(17)):
        net = copy.deepcopy(base_classifier)
        net = net.to(device)
        # print("model recovered to original one")
        console.print("model recovered to original one", style="bold red")
        res.append(ith_shapley_value(17, i, net, test_loader, device, base_classifier))
    log(logfilename, "shapley values:".join('{:.2f}'.format(v) for v in res))




if __name__ == "__main__":
    main()
