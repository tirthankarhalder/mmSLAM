import os
from utils.helper import *
from utils.dataProcess import *
import ast
from datetime import datetime,timedelta
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from model.dataloader import PointCloudDataset
from model.upSamplingBlock import UpSamplingBlock
from model.customLossFunction import CombinedLoss


if __name__ == "__main__":

    preprocess_ExportData(visualization=True)

    

    batch_size = 32


    tensorDataset = PointCloudDataset(total_frameStacked)
    dataloader = DataLoader(tensorDataset, batch_size=batch_size, shuffle=True)

    tensorDatasetGroundTruth = PointCloudDataset(total_frameStacked)
    dataloaderGrounfTruth = DataLoader(tensorDataset, batch_size=batch_size, shuffle=True)



    device = torch.device("cpu")
    
    model = UpSamplingBlock().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = CombinedLoss(alpha=0.5).to(device)
    epochs = 100
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (point_cloud_batch,point_cloud_batchGroundTruth) in enumerate(zip(dataloader, dataloaderGrounfTruth)):
            print(f"Batch {batch_idx+1}:")
            print(f"Shape of point cloud batch: {point_cloud_batch.shape}")  #(32, 1000, 3)

            optimizer.zero_grad()

            UpSamplingBlockWeights,seedGenWwights,noiseAwareFFWeights,confidenseScoreWeights = model(point_cloud_batch)
            loss = criterion([UpSamplingBlockWeights,seedGenWwights,noiseAwareFFWeights,confidenseScoreWeights], point_cloud_batchGroundTruth)#groundTruth (10000,10000,3)
            loss.backward()
            
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 10 == 9:  
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {running_loss / 10:.4f}')
                running_loss = 0.0
