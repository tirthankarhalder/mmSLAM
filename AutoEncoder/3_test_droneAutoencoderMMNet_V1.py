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
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), "..", "/app")))
os.chdir("/app")
from AutoencoderMMNet_V1 import Autoencoder
from DatasetDrone_AllAutoencoder import DatasetDrone
from chamfer_distance import ChamferDistance
import torch.nn.functional as F

from scipy.io import savemat, loadmat
from torch_geometric.loader import DataLoader
#import config as cfg
from Emd.emd_module import emdFunction
from datetime import datetime

from tqdm import tqdm
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
    # processedDataFolder_name = "./processedData/2025-02-11_22-23-15/"
    processedDataFolder_name = os.path.abspath("./processedData/2025-03-18_11-11-03/")
    model_path = processedDataFolder_name + '/dronetrained/checkpoints/2025-03-18T12:00:49.835403/MMNet_ChBest.pt'#'./trained/MMNet_ChBest.pt'
    print(f"Saved Model Path: {model_path}")
    #tensorborad writer
    #writer = SummaryWriter(comment="color_GAN_test")
    #dataset = Completion3D('../data/Completion3D', split='train', categories='Airplane')
    test_dataset = DatasetDrone(processedDataFolder_name + '/droneData_Test', split='test')
    
    test_data_loader = DataLoader(test_dataset, batch_size=1, follow_batch=['y', 'x'],shuffle=False,drop_last=False)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    
    G = Autoencoder(device).to(device)
    
    ChD = ChamferDistance()  # chamfer loss for 

    checkpoint = torch.load(model_path,map_location=device)
    G.load_state_dict(checkpoint['Gen_state_dict'])

    datalistTxt = processedDataFolder_name + "/droneData_Test/datalist.txt"
    with open(datalistTxt, "r") as f:
        mat_file_paths = [line.strip() for line in f.readlines() if line.strip().endswith(".mat")]

    mat_filenames = [path.split("/")[-1] for path in mat_file_paths]
    mat_filenames_array = np.array(mat_filenames)

    parentDir = processedDataFolder_name + "/outputDroneTest/"
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
    loss_gEncoded =0
    each_chd = []
    each_emd = []
    each_chdEncoded = []
    each_emdEncoded = []
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
        Max=1e17;Min=0
        x_ini=(x_ini-Max)/(Max-Min)
        # print(x_ini.shape,x_pos.shape,gt_pcd.shape) #torch.Size([1, 3]) torch.Size([1024, 3]) torch.Size([2048, 3])
        score,ini_points,pred,pred_Doppler,scoreEncoded,pd_pointsEncoded,predDopplerEncoded,ini_pointsEncoded = G(x_ini,x_pos,data.x_batch)
        dist1, dist2, idx1, idx2 = ChD(pred, gt_pcd.view(batch_size,-1,3))  # test G 
        dist1Encoded, dist2Encoded, idx1Encoded, idx2Encoded = ChD(pd_pointsEncoded, x_pos.view(batch_size,-1,3))  # test G 
        # print(score.shape,ini_points.shape,pred.shape,pred_Doppler.shape,scoreEncoded.shape,pd_pointsEncoded.shape,predDopplerEncoded.shape,ini_pointsEncoded.shape)
        g_error = 0.5*(torch.mean(torch.sqrt(dist1))) + 0.5*(torch.mean(torch.sqrt(dist2)))
        g_errorEncoded = 0.5*(torch.mean(torch.sqrt(dist1Encoded))) + 0.5*(torch.mean(torch.sqrt(dist2Encoded)))
        #print(g_error.size())
        loss_g += g_error.item()
        loss_gEncoded += g_errorEncoded.item()
        emd_error = emd(pred,gt_pcd.view(batch_size,-1,3))
        emd_errorEncoded = emd(pd_pointsEncoded,x_pos.view(batch_size,-1,3))
        
        predDopplerEncoded = predDopplerEncoded.squeeze(-1)
        loss_dopplerMse = F.mse_loss(predDopplerEncoded,x_ini)
        gen_data = {
        'input': x_pos.cpu().numpy().reshape((-1,3)),
        'pred_pcd': pred.detach().cpu().numpy().reshape((-1,3)),
        'pred_pcdDecoded': pd_pointsEncoded.detach().cpu().numpy().reshape((-1,3)),
        'gt_pcd': gt_pcd.cpu().numpy().reshape((-1,3)),
        'predDoppler': pred_Doppler.detach().cpu().numpy().reshape((-1,1)),
        'predDopplerDecoded': predDopplerEncoded.detach().cpu().numpy().reshape((-1,1)),
        'Chd':g_error.item(),
        'EMD':emd_error.item(),
        'ChdEncoded':g_errorEncoded.item(),
        'EMDEncoded':emd_errorEncoded.item(),
        'LossDoppler':loss_dopplerMse.item()
        }
        each_chd.append(g_error.item())
        each_emd.append(emd_error.item())
        each_chdEncoded.append(g_errorEncoded.item())
        each_emdEncoded.append(emd_errorEncoded.item())
        
        savemat(folder_path + f"/{mat_filenames_array[step]}", gen_data)
        step = step + 1
    print("loss_g/len(test_dataset): ",loss_g/len(test_dataset))
    save_txt(folder_path + "/chd_loss.txt",np.array(each_chd))
    save_txt(folder_path + "/emd_loss.txt",np.array(each_emd))
    save_txt(folder_path + "/chd_lossEncoded.txt",np.array(each_chdEncoded))
    save_txt(folder_path + "/emd_lossEncoded.txt",np.array(each_emdEncoded))
