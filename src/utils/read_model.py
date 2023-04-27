# -*- coding: utf-8 -*-

import torch
import tensorflow as tf
# import numpy as np
def read_torch_pt(path):
    model_dict = torch.load(path, map_location=torch.device('cpu'))
    
    w = []
    b = []
    name = []
    
    for key in model_dict.keys():
        name.append(key)
        if 'weight' in key:
            w.append(tf.constant(model_dict[key]))
        elif 'bias' in key:
            b.append(tf.constant(model_dict[key]))
        
    return name, w, b

if __name__ == '__main__':
    path = 'model.pt'
    name_list, weight_list, bias_list = read_torch_pt(path)
    
    
    
    
    