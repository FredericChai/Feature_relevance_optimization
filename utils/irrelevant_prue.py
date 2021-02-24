import shutil
import torch
import sys
import os
import sklearn 
import re
import math
import numpy as np
import torchvision.models as models
import torch.nn as nn
from collections import OrderedDict
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
from model import VGG,resnet34 


#hook function
def get_ft(name):
    def hook(model,input,output):
        activation[name] = output.detach()
    return hook

# register forward hook for all layer
def register_submodule(NN):
    NN.eval()
    for i in range(3):
        NN.layer1[i].conv1.register_forward_hook(get_ft('l1c1_b'+str(i)))
        NN.layer1[i].conv2.register_forward_hook(get_ft('l1c2_b'+str(i)))
    # register layer2-4
    for i in range(4):
        NN.layer2[i].conv1.register_forward_hook(get_ft('l2c1_b'+str(i)))
        NN.layer2[i].conv2.register_forward_hook(get_ft('l2c2_b'+str(i))) 
    for i in range(23):
        NN.layer3[i].conv1.register_forward_hook(get_ft('l3c1_b'+str(i)))
        NN.layer3[i].conv2.register_forward_hook(get_ft('l3c2_b'+str(i)))
    for i in range(3):
        NN.layer4[i].conv1.register_forward_hook(get_ft('l4c1_b'+str(i)))
        NN.layer4[i].conv2.register_forward_hook(get_ft('l4c2_b'+str(i)))
    
def cal_centre(dataloaders):
    for batch_idx, (inputs, targets) in enumerate(dataloaders):
        inputs = inputs.numpy()
        if batch_idx ==0:
            img_list = inputs
        elif batch_idx >0:
            img_list = np.append(img_list,inputs,axis=0)
    three_channel_mean = np.mean(img_list,axis=(0))
    centre  =three_channel_mean
    centre = torch.from_numpy(centre)
    # centre = centre.cuda()
    centre_size = centre.shape[1]
    return centre

def get_avg_batch_feature(feature,tem_feature):
    for ft_layer in feature:
        feature[ft_layer] = torch.mean(feature[ft_layer],0)
    for ft_layer in tem_feature:
        tem_feature[ft_layer] += feature[ft_layer]
    return tem_feature

def get_all_activation(NN,dataloaders):
    #intialize a temporal container for recoding feature
    x = torch.randn(4,3,224,224)
    global activation
    activation = {}
    output = NN(x)
    tem_activation = {}
    for i in activation:
        tem_activation[i] = torch.zeros(activation[i].shape[1:])
    activation = {}

    for batch_idx, (inputs, targets) in enumerate(dataloaders):
        torch.no_grad()   
    #     inputs = inputs.cuda()
        out= NN(inputs) #feature was recorded in activation dictionary
        tem_activation = get_avg_batch_feature(activation,tem_activation)

    return tem_activation

def get_filter_rank(tem_ft,tem_centre,rank_list,criteria):
    for ft_layer in tem_ft:
        ft_dist = torch.ones(tem_ft[ft_layer].shape[0])
        i=0
        while tem_centre.shape[-1]>tem_ft[ft_layer].shape[-1]:
            tem_centre = F.avg_pool2d(tem_centre,2)
        t_centre = torch.mean(tem_centre,0)
#         print('tem_ft[ft_layer]:',tem_ft[ft_layer].shape)
        for ft in tem_ft[ft_layer]:
            dist = torch.mean(criteria(ft,t_centre),dim=0)
            ft_dist[i] = dist
            i+=1 # i is to record position of ft: [0.7,0.8...]
            _,rank_list[ft_layer] = torch.sort(ft_dist,descending= True)
        # the returned rank is similarity from low to high.
        # rank[0] means most irrelevant one
    return rank_list

#save all layer
def getkeys():
    l = 0
    cov = 0
    layer_key=[]
    rank_key = []
    for j in [3,4,6,3]:
        l+=1
        for i in range(j):
            #i is the block
            for cov in [1,2]:
                rank_key.append('l{}c{}_b{}'.format(l,cov,i))
                layer_key.append('layer{}.{}.conv{}.weight'.format(l,i,cov))
    return rank_key,layer_key
#rank_key: l1c1b1, layerkey: layer.0.1.weight

def prune_irrelevant(cp_path,NN,dataloaders,ratio,criteria):
    register_submodule(NN)
    tem_activation = get_all_activation(NN,dataloaders)
    rank_list = tem_activation #initialize rank_list with same keys
    if criteria =='dist':
        dist = nn.PairwiseDistance(p=2)
    centre = cal_centre(dataloaders)
    tem_centre = centre
    rank_list = get_filter_rank(tem_activation,tem_centre,rank_list,dist)
    cp_path = cp_path 
    cp = torch.load(cp_path)['state_dict']     
    rank_key,layer_key = getkeys()
    ratio = ratio
    for key_i in range(len(rank_key)):
        length = len(rank_list[rank_key[key_i]])
        threshold = np.round(ratio*length)
    #     print(length,threshold)
        # print(rank_key[key_i])
        # print(layer_key[key_i])
        for i in range(int(threshold)):    
            signal_irrev = rank_list[rank_key[key_i]][i].item()
            print('filter to prune:',signal_irrev)
            zeros = torch.randn(cp[layer_key[key_i]].shape[1:])
            cp[layer_key[key_i]][signal_irrev]=zeros

    return cp