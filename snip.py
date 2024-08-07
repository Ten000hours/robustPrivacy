import imp
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import types

# from snl_finetune_unstructured import LearnableAlpha

def snip_forward_conv2d(self, x):
        return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)

def snip_forward_relu(self, x):
        return F.relu(x * self.weight_mask, inplace=False)
                        

def snip_forward_linear(self, x):
        return F.linear(x, self.weight * self.weight_mask, self.bias)


def SNIP(net, keep_ratio, train_dataloader, device):
    # TODO: shuffle?

    # Grab a single batch from the training dataset
    inputs, targets = next(iter(train_dataloader))
    inputs = inputs.to(device)
    targets = targets.to(device)

    # Let's create a fresh copy of the network so that we're not worried about
    # affecting the actual training-phase
    net = copy.deepcopy(net)

    # Monkey-patch the Linear and Conv2d layer to learn the multiplicative mask
    # instead of the weights
    for name, param in net.named_parameters():
        if 'alpha' in name:
            param.requires_grad = True

        # Override the forward methods:
        # if isinstance(layer, nn.Conv2d):
        #     layer.forward = types.MethodType(snip_forward_conv2d, layer)

        # if isinstance(layer, nn.Linear):
        #     layer.forward = types.MethodType(snip_forward_linear, layer)

        # if isinstance(layer, nn.ReLU):
        #     layer.forward = types.MethodType(snip_forward_relu, layer)

    # Compute gradients (but don't apply them)
    net.zero_grad()
    outputs = net.forward(inputs)
    loss = F.nll_loss(outputs, targets)
    loss.backward()

    grads_abs = []
    for name, param in net.named_parameters():
        if 'alpha' in name:
            grads_abs.append(torch.abs(param.grad))
    print("len of grad: ", len(grads_abs))

    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in grads_abs])
    norm_factor = torch.sum(all_scores)
    all_scores.div_(norm_factor)


# =============================================grad allocation=======================
    budget_list= []
    for layer_grads in grads_abs:
        layer_scores = torch.tensor(layer_grads)
        layer_total_score = torch.sum(layer_scores)
        layer_total_score.div_(norm_factor)
        # print('layer score: ', layer_total_score, ' and len(layergrads): ',len(layer_grads))
        # print(layer_grads)
        layer_budget= int((layer_total_score) * int(len(all_scores) * keep_ratio))
        budget_list.append(min(torch.numel(layer_scores) ,layer_budget))
# =============================================grad allocation=======================

# ================================================shapley ==========================================
    # budget_list = []
    # shapley = torch.FloatTensor([1.75,0.40,2.09,2.34,2.36,2.99,2.73,0.33,2.01,2.71,2.47,0.69,0.32,0.26,0.00,0.01])
    # total = torch.sum(shapley)
    # shapley.div_(total)
    # for i,value in enumerate(shapley):
    #     layer_budget= int((value) * int(len(all_scores) * keep_ratio))
    #     budget_list.append(min(torch.numel(grads_abs[i]) ,layer_budget))
# ================================================shapley ==========================================   
        


    # num_params_to_keep = int(len(all_scores) * keep_ratio)
    # threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
    # acceptable_score = threshold[-1]
    

    # keep_masks = []
    # for g in grads_abs:
    #     keep_masks.append(((g / norm_factor) >= acceptable_score).float())

    # print('total remained params: ',torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks])))

    # return(keep_masks)
    return(budget_list)
