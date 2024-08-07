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
from random import sample

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
parser.add_argument('--threshold', default=1e-2, type=float)
# parser.add_argument('--gpus', default=1, type=int,
#                     help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--logname', type=str, default='log.txt')
# parser.add_argument('--print-freq', default=100, type=int,
#                     metavar='N', help='print frequency (default: 10)')
parser.add_argument('--stride', type=int, default=1, help='conv1 stride')
args = parser.parse_args()

layer_mask = 1


def find_subsets(nums):
    subsets = []
    # dic = {}
    for i in range(len(nums)-4, len(nums)):
        # subsets = []
        subsets.extend(itertools.combinations(nums, i))
        # sub = itertools.combinations(nums, i)
        # dic[i] = [list(subset) for subset in subsets] 
    # print([list(subset) for subset in subsets])
    return [list(subset) for subset in subsets]
    # return dic


def relu_counting(net, args):
    relu_count = 0
    for name, param in net.named_parameters():
        if 'alpha' in name:
            boolean_list = param.data > args.threshold
            relu_count += (boolean_list == 1).sum()
    return relu_count

def mask_model(model, test_loader, device, mask):
    for name, layer in model.named_children():
        if isinstance(layer, nn.Sequential) or isinstance(layer, BasicBlock_IN):
            mask_model(layer, test_loader, device, mask)
        elif isinstance(layer, LearnableAlpha):
            # print('name: ',name, 'and layer:',layer)
            global layer_mask
            if layer_mask not in mask:
                layer.alphas.data = torch.zeros_like(layer.alphas)
                # print("layer ", layer_mask, "is masked")
            layer_mask += 1
    # return model

def mask_model_wrn(model, test_loader, device, mask):
    for name, param in model.named_parameters():
        if "alpha" in name:
            # print('name: ',name, 'and layer:',layer)
            global layer_mask
            if layer_mask not in mask:
                param.data = torch.zeros_like(param.data)
                # print("layer ", layer_mask, "is masked")
            layer_mask += 1

def get_utility( test_loader, device, mask, base_classifier, type):
    # print("==========================================")
    # tempac = model_inference(model, test_loader, device, display=True)

    # print("acc before mask: ", tempac)
    # print("==========================================")
    model = copy.deepcopy(base_classifier)
    model = model.to(device)
    # remove the relu layers based on mask 
    global layer_mask 
    layer_mask = 1
    if type == 'wrn':
        mask_model_wrn(model, test_loader, device, mask )
    else: 
        mask_model(model, test_loader, device, mask )

    return model_inference(model, test_loader, device, display=False)

from scipy.special import comb, perm
# #计算排列数
# A=perm(3,2)
# #计算组合数
# C=comb(45,2)
# print(A,C)

from collections import deque
def ith_shapley_value(subsets, monteCarloSample, relu_layers_No, layer_index, model, test_loader, device, base_model,type):
    # monteCarloSample = sample([i for i in range(1, relu_layers_No+1) if i != layer_index], subsetNum)
    # subsets = find_subsets(monteCarloSample)
    sv =0 
    for mask in subsets[::-1]:
        U_s = get_utility(test_loader, device, mask, base_model, type)
        all_mask = mask +[layer_index]
        U_si = get_utility( test_loader, device, all_mask, base_model, type)
        if U_si < 7:
            break
# =======================tree search==================================
    # qu = deque()    
    # for subset in subsets:
    #     qu.append(subset)
    # # for index, masks in subsets.items():      
    # while len(qu) != 0:
    #     print("queue length: ", len(qu))  
    #     mask= qu.popleft()
    #     try:
    #         U_s = get_utility(test_loader, device, mask, base_model, type)
    #         all_mask = mask +[layer_index]
    #         U_si = get_utility(test_loader, device, all_mask, base_model, type)
    #     except Exception as e:
    #         print("mask : ", mask)
    #     # early truncation       
    #     if U_si > 15:
    #         print("mask sub: ", mask)
    #         for child in find_subsets(mask):
    #             if child not in qu:
    #                 qu.append(child)
# =======================tree search==================================
        # =================exact sv calculating ================            
        sv += (U_si - U_s)/comb(relu_layers_No-1,len(mask))
    sv = sv * 1/relu_layers_No
    # ==============================================

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

    lenRelu = 0
    for name, param in net.named_parameters():
     # param.requires_grad = False
        if 'alpha' in name:
            # param.requires_grad = True
            lenRelu +=1
    print("total # of relu layers = ", lenRelu)

# shapley values (perm): [0.0566337425595238, 0.09678317212301589, 0.14673694816468255, -0.01661306423611111, -0.0316541728670635, 0.012727182539682545, 0.09678673735119049, -0.02139081101190476]

# shapley value (comb):[10.981988095238092, 11.128154761904765, 11.550011904761906, 11.034011904761893, 10.8500119047619, 10.87782142857142, 6.478488095238097, 0.13951190476190362]

# =====================================================test code============================================================
    global layer_mask 
    layer_mask = 1
    mask = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,16]
    print(mask)
    net = copy.deepcopy(base_classifier)
    # mask_model(net, test_loader, device, mask)
    mask_model_wrn(net, test_loader, device, mask)
    acc=model_inference(net, test_loader, device, display=True)
    print('relu count: ',relu_counting(net, args))
    print(acc)
# ======================================================================================================
    # # log(logfilename, "Loaded the base_classifier")

    # # Calculating the loaded model's test accuracy.
    original_acc = model_inference(base_classifier, test_loader,
                                    device, display=True)
    # print(original_acc)
    # log(logfilename, "Original Model Test Accuracy: {:.5}".format(original_acc))

    # Creating a fresh copy of network not affecting the original network.

    net = net.to(device)
    # print(net)

    res = []
    t = 0
    phi = [0] * 8
    phi_ = [0] * 8
    flag = False
    # while abs(sum(phi) - sum(phi_))< 6:
    # =============================attempt of data shapley=======================================
    # resnet9 shapley values:8.85,0.02,0.23,-0.17,-0.10,-0.03,-3.32,-5.36

    # for epoch in range(10):
    #     console.print("abs difference: ", abs(sum(phi) - sum(phi_)),style="bold red")
    #     phi_ = phi
    #     t += 1
    #     v_j, v_j1 = 0,0
    #     monteCarloSample = sorted(sample([i for i in range(1, 8+1)], 8))
    #     console.print('random samples: ', monteCarloSample)
    #     for i in tqdm(range(1,9)):
    #         net = copy.deepcopy(base_classifier)
    #         net = net.to(device)
    #         # print("model recovered to original one")
    #         console.print("model recovered to original one", style="bold red")
    #         if i in monteCarloSample:
    #             monteCarloSample.remove(i)
    #             flag = True
    #         else:
    #             flag = False
    #         if original_acc - v_j1 < 5:
    #             v_j = v_j1
    #         else:
    #             console.print('removed samples: ', monteCarloSample)
    #             v_j = ith_shapley_value(monteCarloSample, 8, i, net, test_loader, device, base_classifier)
    #         phi[i-1] = (t-1)/t * phi_[i-1] + (v_j - v_j1)/t
    #         v_j1 = v_j
    #         if flag == True: 
    #             monteCarloSample.append(i)
    #         monteCarloSample = sorted(monteCarloSample)
    # log(logfilename, "shapley values:" + ",".join('{:.2f}'.format(v) for v in phi))
    # ==========================================data shapley====================================
    # ======================================only early truncation ================================


# Cifar100
    # resnet9 shapley values:8.85,8.87,9.10,8.93,8.83,8.79,5.47,0.11
    # resnet18 shapley values:1.77,1.17,4.39,3.67,4.81,4.84,4.94,2.00,4.94,4.88,4.93,1.55,4.40,4.88,1.57,0.01
    # subset = 2^5 || resnet34 shapley values:0.27,0.46,1.34,0.36,2.79,1.68,2.59,0.22,2.13,0.13,2.31,0.09,3.06,0.15,2.74,2.25,2.96,0.14,2.96,0.21,2.97,0.31,2.97,0.03,2.95,0.15,1.50,2.97,2.88,0.23,0.14
    # subset = 2^9 || resnet34 shapley values: 0.27,0.46,1.34,0.36,2.79,1.68,2.59,0.22,2.13,0.13,2.31,0.09,3.06,0.15,2.74,2.25,2.96,0.14,2.96,0.21,2.97,0.31,2.97,0.03,2.95,0.15,1.50,2.97,2.88,0.23,0.1


    # wrn shapley values:0.63,1.99,3.20,1.34,1.99,3.23,5.26,5.30,5.31,4.91,5.25,5.26,5.29,5.28,5.26,5.30,2.75,0.33
    # wrn tree shapley values:0.37,0.98,1.28,0.74,0.77,1.17,1.72,1.74,1.75,1.58,1.72,1.69,1.71,1.72,1.72,1.73,0.84,0.17

# Cifar10
    #resnet18 t=40 shapley values:0.14,-0.02,0.40,0.46,0.45,0.49,0.48,0.08,0.44,0.49,0.47,0.18,0.13,0.09,0.01,0.00
    # resnet18 t=20 shapley values:0.27,0.08,0.45,0.53,0.58,0.60,0.68,0.07,0.59,0.67,0.64,0.28,0.10,0.18,0.04,0.00
    #          t=10 :  1.75,0.40,2.09,2.34,2.36,2.99,2.73,0.33,2.01,2.71,2.47,0.69,0.32,0.26,0.00,0.01



    # monteCarloSample = sorted(sample([i for i in range(1, lenRelu +1)], lenRelu))
    # # print(find_subsets(monteCarloSample))
    # console.print('random samples: ', monteCarloSample)
    # type ='n'
    # for i in tqdm(range(1,lenRelu+1)):
    #     net = copy.deepcopy(base_classifier)
    #     net = net.to(device)
    #     # print("model recovered to original one")
    #     console.print("model recovered to original one", style="bold red")
    #     if i in monteCarloSample:
    #         monteCarloSample.remove(i)
    #         flag = True
    #     else:
    #         flag = False
    #     console.print('removed samples: ', monteCarloSample)
    #     subsets = find_subsets(monteCarloSample)
    #     # console.print('subsets ', subsets)
    #     v_j = ith_shapley_value(subsets,monteCarloSample, lenRelu, i, net, test_loader, device, base_classifier, type)
    #     # phi[i-1] = (t-1)/t * phi_[i-1] + (v_j - v_j1)/t
    #     # v_j1 = v_j
    #     if flag == True: 
    #         monteCarloSample.append(i)
    #     monteCarloSample = sorted(monteCarloSample)
    #     res.append(v_j)
    # log(logfilename, "shapley values:" + ",".join('{:.2f}'.format(v) for v in res))
    # ==========================================early truncation===========================================




if __name__ == "__main__":
    main()
