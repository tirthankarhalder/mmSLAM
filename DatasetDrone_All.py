#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 10:07:17 2021
UofSC indoor point cloud Data set processing
@author: pingping
"""
import os
from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_zip)
from scipy.io import savemat, loadmat
import torch
import open3d
import numpy as np

from dataclasses import dataclass

## IMPORTANT only for python version > 3.7
@dataclass
class PCD:
    x: float
    y: float
    z: float = 0.0
    
class DatasetDrone(torch.utils.data.dataset.Dataset):
    
    def __init__(self, txt_path, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        self.scene_list = read_list(txt_path+'/datalist.txt',split)
        self.txt_path=txt_path
        self.all_data = get_all_item(self.scene_list,txt_path)
        
    def __len__(self):
        return len(self.scene_list)

    def __getitem__(self, idx):    
        data = self.all_data[idx]
        #data = get_item(self.scene_list,self.txt_path,idx)
        return data

def read_list(processed_path,split):
    '''
    token shape: 'data_path'
    '''
    file_list = []
    #print(processed_path)
    with open(processed_path, 'r') as f:
        count = 0
        for line in f.readlines():
#            count = count + 1
#            if count > 9:
#               break
            token = line.strip('\n').split('\t')
            file_list.append(token[0])
    return file_list

def get_item(scene_list,data_path,idx):
    '''
    '''
    path = scene_list[idx]
    in_path = path.lstrip('..')
    
    #load gt pcd and incomplete pcd
    mat_file = data_path + in_path
    #print(mat_file)
    mat = loadmat(mat_file)
    gt_cloud = mat['depthPCD']
    #resampling gt_cloud
    b_idx = np.random.choice(gt_cloud.shape[0],4096,replace=True)
    gt_cloud = gt_cloud[b_idx,:]
    ini_pt = mat['radarPCD'].T
    ini_pt = ini_pt.T
    b_idx = np.random.choice(ini_pt.shape[0],1024,replace=True)
    in_cloud = ini_pt[b_idx,:]
    #pt_cloud = pt_cloud[0,0]
    #radar_pose = mat['pose']
    #radar_pose = radar_pose[0,0]
    #rot_ang = radar_pose[:,3:6]
    #pos = radar_pose[:,0:3]
    # ini_pos = mat['ini_pose']
    ini_pos = np.array([[0.0,  0.0, 0.0]])
    #pos = (pos-ini_pos)/4
#    sar_imgs = mat['sar_imgs']
#    fft_data = mat['fft']  #M:256 N:9000       H:4  W:3
#    fft_data = np.absolute(fft_data)
    gt_cloud = torch.from_numpy(gt_cloud).float()
    ini_pt = torch.from_numpy(in_cloud).float()
    radar_ini = torch.from_numpy(ini_pos).float()
    #radar_pos = torch.from_numpy(pos).float()
    #radar_ang = torch.from_numpy(rot_ang).float()
#    radar_imgs = torch.from_numpy(sar_imgs).float()
#    fft_data = torch.from_numpy(fft_data[:,:,1,1]).float()
#    fft_data = fft_data.T
    data=Data(ini=radar_ini,x=ini_pt, y=gt_cloud)
    return data

def get_all_item(scene_list,data_path):
    '''
    '''
    all_data = []
    for idx in range(len(scene_list)):
        path = scene_list[idx]
        # in_path = path.lstrip('..')

        #load gt pcd and incomplete pcd
        # mat_file = data_path + in_path
        mat_file = path
        #print(mat_file)
        mat = loadmat(mat_file)
        gt_cloud = mat['depthPCD']
        # print(f"gt_cloud.shape before: {gt_cloud.shape}")
        #resampling gt_cloud
        b_idx = np.random.choice(gt_cloud.shape[0],16384,replace=True)
        gt_cloud = gt_cloud[b_idx,:]
        # print(f"gt_cloud.shape after: {gt_cloud.shape}")
        ini_pt = mat['radarPCD'].T
        #resampling input point_cloud
        ini_pt = ini_pt.T
        b_idx = np.random.choice(ini_pt.shape[0],1024,replace=True)
        in_cloud = ini_pt[b_idx,:]

        # ini_pos = mat['ini_pose']
        ini_pos = np.array([[0.0,  0.0, 0.0]])
        gt_cloud = torch.from_numpy(gt_cloud).float()
        ini_pt = torch.from_numpy(in_cloud).float()
        radar_ini = torch.from_numpy(ini_pos).float()
        
        all_data.append(Data(ini=radar_ini,x=ini_pt, y=gt_cloud))
        
    return all_data
   
def collate_wrap(batch):
    '''
    costom collate wrapper
    '''
    data = []
    batch_index = 0
    for pt, gt,pn in batch:
        ## gen batch indicator for each points
        patch_vec = torch.zeros(pt.size(0),dtype=torch.int64) + batch_index
        data += [Data(pos=pt, y=gt, patch=patch_vec, normal = pn)]
        batch_index = batch_index + 1
    return InMemoryDataset.collate(data)


#dataset = DatasetUofSC('../Data', split='train')
#dataset[1]
