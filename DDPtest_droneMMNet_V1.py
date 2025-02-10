#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun jan  3 16:04:51 2025
Test ColorGAN.py
@author: tirtha

"""

import torch
import torch.nn as nn
import numpy as np
from MMNet_V1 import Generator
from DatasetDrone import DatasetDrone
from chamfer_distance import ChamferDistance
from scipy.io import savemat, loadmat
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
#import config as cfg
from Emd.emd_module import emdFunction
from datetime import datetime
import os
from tqdm import tqdm
from collections import OrderedDict
def save_txt(path,pred_pcd):
    '''
    pred_pcd: N by 3
    '''
    np.savetxt(path + '.txt', pred_pcd, fmt='%.6f')
    
def emd(p1,p2):
    emdist, _ = emdFunction.apply(p1, p2, 0.01, 500)
    return torch.sqrt(emdist).mean()

if __name__ == '__main__':
    #change the file name and run
    processedDataFolder_name = "./processedData/2025-02-05_13-38-22/"
    #tensorborad writer
    #writer = SummaryWriter(comment="color_GAN_test")
    #dataset = Completion3D('../data/Completion3D', split='train', categories='Airplane')
    test_dataset = DatasetDrone(processedDataFolder_name + 'droneData_Test', split='test')
    
    test_data_loader = DataLoader(test_dataset, batch_size=1, follow_batch=['y', 'x'],shuffle=False,drop_last=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    
    G = Generator().to(device)
    
    ChD = ChamferDistance()  # chamfer loss for 

    checkpointPath = processedDataFolder_name + "dronetrained/checkpoints/"
    listChekpointsFolder = os.listdir(checkpointPath)
    sortedListChekpointsFolder = sorted(listChekpointsFolder, key=lambda x: datetime.fromisoformat(x))
    ddpCheckpointFolder = sortedListChekpointsFolder[-2:]
    print(ddpCheckpointFolder)
    ddpCheckpointFiles = [os.path.join(checkpointPath, d, "MMNet_ChBest.pt") for d in ddpCheckpointFolder]
    model_path = checkpointPath+ f"{sortedListChekpointsFolder[-1]}/MMNet_ChBest.pt"  
    checkpoint = torch.load(ddpCheckpointFiles[0], map_location="cuda")
    if "Gen_state_dict" in checkpoint:
        state_dict = checkpoint["Gen_state_dict"]  # Extract model state dict
    else:
        state_dict = checkpoint
    # model_path = processedDataFolder_name + "dronetrained/checkpoints/2025-02-06T13:18:50.727110/MMNet_ChBest.pt"
    # checkpoint = torch.load(model_path)
    # state_dict = checkpoint['Gen_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")  # Remove "module." prefix
        new_state_dict[new_key] = v

    # new_state_dict_fixed = OrderedDict()
    # for k, v in new_state_dict.items():
    #     new_k = k.replace("pn1_conv", "pn1_module.conv").replace("pn2_conv", "pn2_module.conv").replace("gn1_nn", "gn1_module.nn")
    # new_state_dict_fixed[new_k] = v
    G.load_state_dict(new_state_dict)

    datalistTxt = processedDataFolder_name + "droneData_Test/datalist.txt"
    with open(datalistTxt, "r") as f:
        mat_file_paths = [line.strip() for line in f.readlines() if line.strip().endswith(".mat")]

    mat_filenames = [path.split("/")[-1] for path in mat_file_paths]
    mat_filenames_array = np.array(mat_filenames)

    parentDir = processedDataFolder_name + "outputDroneTest/"
    # current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    timeDir = processedDataFolder_name.split("/")[-2]
    #enable it when multiple test scenios required
    # folder_path = os.path.join(parentDir, timeDir)
    folder_path = parentDir
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")

    G.eval()
    step = 0
    # print ('Valid: ')
    loss_g =0
    each_chd = []
    each_emd = []
    pbar = tqdm(test_data_loader, desc="Testing Progress")

    for data in pbar:
        # print("data shape: ", len(data))
        # print(f"File name: {mat_filenames_array[step]} generated.")
        file_name = mat_filenames_array[step]
        pbar.set_postfix(file=file_name)
        data =data.to(device)
        # 1. Test G 
        gt_pcd = data.y     # 10000 by 3
        #x_fft = data.fft   #N:9000 M:256 H:4  W:3
        x_pos = data.x   #N:9000 M:3
        #x_img = data.imgs   #N:  M: 1   H: 80 W:80
        #x_ang = data.ang      #N:9000 M:3
        x_ini = data.ini
        batch_size = torch.max(data.y_batch)+1
        # print(x_ini.shape,x_pos.shape,gt_pcd.shape) #torch.Size([1, 3]) torch.Size([1024, 3]) torch.Size([2048, 3])
        score,init,pred = G(x_ini,x_pos,data.x_batch)
        dist1, dist2, idx1, idx2 = ChD(pred, gt_pcd.view(batch_size,-1,3))  # test G 

        g_error = 0.5*(torch.mean(torch.sqrt(dist1))) + 0.5*(torch.mean(torch.sqrt(dist2)))
        #print(g_error.size())
        loss_g += g_error.item()
        emd_error = emd(pred,gt_pcd.view(batch_size,-1,3))
        gen_data = {
        'input': x_pos.cpu().numpy().reshape((-1,3)),
        'pred_pcd': pred.detach().cpu().numpy().reshape((-1,3)),
        'gt_pcd': gt_pcd.cpu().numpy().reshape((-1,3)),
        'Chd':g_error.item(),
        'EMD':emd_error.item(),
        }
        each_chd.append(g_error.item())
        each_emd.append(emd_error.item())
        
        savemat(folder_path + f"/{mat_filenames_array[step]}", gen_data)
        step = step + 1
    print("loss_g/len(test_dataset): ",loss_g/len(test_dataset))
save_txt(folder_path + "/chd_loss.txt",np.array(each_chd))
save_txt(folder_path + "/emd_loss.txt",np.array(each_emd))
