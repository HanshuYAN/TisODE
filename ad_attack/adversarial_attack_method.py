import os
import sys
import torch
import copy
import numpy as np
import random
import time
import argparse
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import pdb
import progressbar

from torch.utils.data import DataLoader
from attk.cw_black import BlackBoxL2
from PIL import Image
from attk.generate_gradient import generate_gradient

from utils import Logger

class Mod(nn.Module):
    def __init__(self,model):
        super(Mod, self).__init__() 

        self.net = model
    def forward(self,x):
        return self.net(x)[0]


def adversarial_attack(count,model,test_loader,epsilon,learning_rate,maxiter,isimgnet=False,iscifar=False,device=torch.device("cuda"),is_tiv=False,isshuffle=True,isblackbox=False):

    #setup random seed 
    random.seed(1216)
    np.random.seed(1216)
    torch.manual_seed(1216)

    # prepare process bar 
    bar = progressbar.ProgressBar(maxval=count, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    # mod to store tiv model 
    if is_tiv:
        model = Mod(model)
    if maxiter == 1:
        maxiter += 1
    generate_grad = generate_gradient(
            device,
            targeted = False,
            )
    attack = BlackBoxL2(
            epsilon,
            learning_rate,
            targeted = False,
            max_steps = maxiter ,
            search_steps = 1,
            cuda = True,
            debug=False,
            isimgnet = isimgnet,
            iscifar = iscifar,
            isblackbox = isblackbox
            )
    
    img_no = 0
    total_success = 0
    l2_total = 0.0
    avg_step = 0
    avg_time = 0
    avg_que = 0
    meta_model = None
    current_index = 0  # count how many sample tested 
    if isimgnet and count ==3000:
        count -= 25
    # add random selection to pick test samples
    number_all = count + 25
    number_finetune = 25
    if isshuffle:
        indice_all = np.random.choice(len(test_loader.dataset), number_all, False)
        indice_attack = indice_all[number_finetune:]
    else:
        indice_attack = np.arange(count) 

    for i, (img, target) in enumerate(test_loader):
        if current_index > count + 1:
            break
        if not i in indice_attack:
            continue 
        current_index += 1
        bar.update(current_index)
        img, target = img.to(device), target.to(device)
        pred_logit = model(img)
        
        pred_label = pred_logit.argmax(dim=1)
        if pred_label != target and isimgnet:
            continue
            
        img_no += 1
        timestart = time.time()
        meta_model_copy = copy.deepcopy(meta_model)
        
        queries = 0
        adv, const, first_step = attack.run(model, meta_model_copy, img, target, i)
        timeend = time.time()

        if len(adv.shape) == 3:
            adv = adv.reshape((1,) + adv.shape)
        adv = torch.from_numpy(adv).permute(0, 3, 1, 2).cuda()
        diff = (adv-img).cpu().numpy()
        l2_distortion = np.sum(diff**2)**.5
        adv_pred_logit = model(adv)
        adv_pred_label = adv_pred_logit.argmax(dim = 1)
        
        success = False
        if adv_pred_label != target:
            success = True
        if l2_distortion > 20:
            success = False
        if success:
            total_success += 1
            l2_total += l2_distortion
            avg_step += first_step
            avg_que += (first_step-1)//5*256+first_step
            avg_time += timeend - timestart
        if total_success == 0:
            pass
        # else:
    # print("[STATS][L1] total = {}, seq = {}, time = {:.3f}, success = {}, distortion = {:.5f}, avg_step = {:.5f}, avg_query = {:.5f},success_rate = {:.3f}".format(img_no, i, avg_time / total_success, success, l2_total / total_success, avg_step / total_success, avg_que / total_success, total_success / float(img_no)))
    # sys.stdout.flush()
    print("\n")
    asr = total_success / float(img_no)
    return 1 - asr

if __name__ == "__main__":
    pass