import torch
import torch.nn.functional as F
from layer_shapley_refactored import *

import copy

# from snl_finetune_unstructured import LearnableAlpha

def snip_forward_conv2d(self, x):
        return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)

def snip_forward_relu(self, x):
        return F.relu(x * self.weight_mask, inplace=False)
                        

def snip_forward_linear(self, x):
        return F.linear(x, self.weight * self.weight_mask, self.bias)


def SNIP(net, keep_ratio, train_dataloader, device, type):

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


    if type == 'senet':
        budget_list= []
        for layer_grads in grads_abs:
            layer_scores = torch.tensor(layer_grads)
            layer_total_score = torch.sum(layer_scores)
            layer_total_score.div_(norm_factor)
            # print('layer score: ', layer_total_score, ' and len(layergrads): ',len(layer_grads))
            # print(layer_grads)
            layer_budget= int((layer_total_score) * int(len(all_scores) * keep_ratio))
            budget_list.append(min(torch.numel(layer_scores) ,layer_budget))

    elif type == 'privshap':
        budget_list = []
        # Use the new helper function to compute layer Shapley values
        # Create a simple test loader from the training batch
        from torch.utils.data import TensorDataset, DataLoader
        test_dataset = TensorDataset(inputs, targets)
        test_loader = DataLoader(test_dataset, batch_size=inputs.size(0), shuffle=False)
        
        # Compute Shapley values for all layers (ordered by layer index)
        shapley_values = compute_layer_shapley_values(net, device, test_loader)
        shapley_tensor = torch.FloatTensor(shapley_values)
        
        total = torch.sum(shapley_tensor)
        shapley_tensor.div_(total)
        for i,value in enumerate(shapley_tensor):
            layer_budget= int((value) * int(len(all_scores) * keep_ratio))
            budget_list.append(min(torch.numel(grads_abs[i]) ,layer_budget))

    return(budget_list)
