import numpy as np
import os
import torch
from models.hrnet import hrnet
from torch import nn


def preloader():
    current_path = os.path.dirname(os.path.abspath(__file__))
    
 
    model = create_model(cfg)
    # if torch.cuda.is_available():
    #     model.cuda()
    model = nn.DataParallel(model)
    # print(torch.load(seg_weights,map_location=torch.device('cpu')).keys())
    model.load_state_dict(torch.load(seg_weights,map_location=torch.device('cpu'))["state_dict"])
    model.eval()
    
    return model