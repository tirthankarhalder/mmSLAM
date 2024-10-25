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
from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

startProcessingForNewData = False
if __name__ == "__main__":


    if startProcessingForNewData:
        print("Start Processing New Data")
        pointcloudRadarDepth = preprocess_ExportData(visualization=False)
    else:
        print("Importing Saved Data")
        pointcloudRadarDepth = pd.read_pickle("./mergedRadarDepth.pkl")

    pointcloudRadarDepth.reset_index(drop=True, inplace=True)


    pointcloudRadarDepth["sampleDepth"] = None
    downsample_size = 2000  # Specify the desired downsampled size
    # downsampled_pcd = np.empty((pointcloudRadarDepth.shape[0], downsample_size, 3))
    for index,row in pointcloudRadarDepth.iterrows():
        # print(index)
        indices = np.random.choice(307200, downsample_size, replace=False)  # Random indices
        # pointcloudRadarDepth.loc[index, "sampleDepth"] = pointcloudRadarDepth["depthPCD"].iloc[index][indices].tolist()  # Select points using the random indices
        pointcloudRadarDepth.at[index, "sampleDepth"] = pointcloudRadarDepth["depthPCD"].iloc[index][indices]



    # pointcloudRadarDepth["sampleDepth"]
    for index,row in pointcloudRadarDepth.iterrows():
        distancesRadar = np.linalg.norm(pointcloudRadarDepth["radarPCD"][index], axis=1)
        normalized_distances = (distancesRadar - distancesRadar.min()) / (distancesRadar.max() - distancesRadar.min())
        sns.set(style="whitegrid")
        fig = plt.figure(figsize=(12,6))
        ax1 = fig.add_subplot(131,projection='3d')
        distancesRadar = np.linalg.norm(pointcloudRadarDepth["radarPCD"][index], axis=1)
        normalized_distancesRadar = (distancesRadar - distancesRadar.min()) / (distancesRadar.max() - distancesRadar.min())
        img1 = ax1.scatter(pointcloudRadarDepth["radarPCD"][index][:, 0], pointcloudRadarDepth["radarPCD"][index][:, 1], pointcloudRadarDepth["radarPCD"][index][:, 2], c=normalized_distancesRadar,cmap = 'viridis', marker='o')
        fig.colorbar(img1)
        ax1.set_title('Point Cloud 1')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        ax2 = fig.add_subplot(132,projection='3d')
        distancesDepth = np.linalg.norm(pointcloudRadarDepth["depthPCD"][index], axis=1)
        normalized_distancesDepth = (distancesDepth - distancesDepth.min()) / (distancesDepth.max() - distancesDepth.min())
        img2 = ax2.scatter(pointcloudRadarDepth["depthPCD"][index][:, 0], pointcloudRadarDepth["depthPCD"][index][:, 1], pointcloudRadarDepth["depthPCD"][index][:, 2], c=normalized_distancesDepth, cmap = 'viridis',marker='o')
        fig.colorbar(img2)
        ax2.set_title('Point Cloud 2')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')

        ax3 = fig.add_subplot(133,projection='3d')
        distancesSampleDepth = np.linalg.norm(pointcloudRadarDepth["sampleDepth"][index], axis=1)
        normalized_distancesSampleDepth = (distancesSampleDepth - distancesSampleDepth.min()) / (distancesSampleDepth.max() - distancesSampleDepth.min())
        img3 = ax3.scatter(pointcloudRadarDepth["sampleDepth"][index][:, 0], pointcloudRadarDepth["sampleDepth"][index][:, 1], pointcloudRadarDepth["sampleDepth"][index][:, 2], c=normalized_distancesSampleDepth, cmap = 'viridis',marker='o')
        fig.colorbar(img3)
        ax3.set_title('Point Cloud 3')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')


        plt.tight_layout()
        plt.savefig("./visualization/RadarDepth/radarDepth_"+ str(index)+".png")
        # plt.show()
        plt.close()
        if index ==3:
            break
    print("Sample Visulization Saved")
    # total_frameStackedRadar = np.random.rand(1000, 1000, 3)
    # total_frameStackedDepth = np.random.rand(1000, 307200, 3)

    total_frameStackedRadar = np.stack(pointcloudRadarDepth["radarPCD"])
    total_frameStackedDepth = np.stack(pointcloudRadarDepth["sampleDepth"])

    print("total_frameStackedRadar.shape: ",total_frameStackedRadar.shape)
    print("total_frameStackedDepth.shape: ",total_frameStackedDepth.shape)


    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    # device = torch.device("cpu")   
    train_radar, temp_radar, train_depth, temp_depth = train_test_split(total_frameStackedRadar, total_frameStackedDepth, test_size=0.3, random_state=42)
    val_radar, test_radar, val_depth, test_depth = train_test_split(temp_radar, temp_depth, test_size=0.5, random_state=42)  # 15% test, 15% validation

    # Create PointCloudDataset for each split
    train_dataset_radar = PointCloudDataset(train_radar)
    train_dataset_depth = PointCloudDataset(train_depth)

    val_dataset_radar = PointCloudDataset(val_radar)
    val_dataset_depth = PointCloudDataset(val_depth)

    test_dataset_radar = PointCloudDataset(test_radar)
    test_dataset_depth = PointCloudDataset(test_depth)

    # Create DataLoader for each split
    train_loader_radar = DataLoader(train_dataset_radar, batch_size=batch_size, shuffle=False, num_workers=4)
    train_loader_depth = DataLoader(train_dataset_depth, batch_size=batch_size, shuffle=False, num_workers=4)

    val_loader_radar = DataLoader(val_dataset_radar, batch_size=batch_size, shuffle=False, num_workers=4)
    val_loader_depth = DataLoader(val_dataset_depth, batch_size=batch_size, shuffle=False, num_workers=4)

    test_loader_radar = DataLoader(test_dataset_radar, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader_depth = DataLoader(test_dataset_depth, batch_size=batch_size, shuffle=False, num_workers=4)


    # tensorDataset = PointCloudDataset(total_frameStackedRadar)#no split
    # dataloader = DataLoader(tensorDataset, batch_size=batch_size, shuffle=False)

    # tensorDatasetGroundTruth = PointCloudDataset(total_frameStackedDepth)
    # dataloaderGrounfTruth = DataLoader(tensorDatasetGroundTruth, batch_size=batch_size, shuffle=False)

    model = UpSamplingBlock().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = CombinedLoss(alpha=0.5).to(device)
    epochs = 100
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (point_cloud_batch,point_cloud_batchGroundTruth) in enumerate(zip(train_loader_radar,train_loader_depth)):
            print(f"Batch {batch_idx+1}:")
            if batch_idx==1:
                print(f"Shape of point cloud batch: {point_cloud_batch.shape}")  #(32, 1000, 3)
                print(f"Shape of point point_cloud_batchGroundTruth: {point_cloud_batchGroundTruth.shape}")  #(32, 1000, 3)
            point_cloud_batch,point_cloud_batchGroundTruth = point_cloud_batch.to(device),point_cloud_batchGroundTruth.to(device)
            optimizer.zero_grad()

            UpSamplingBlockWeights,seedGenWwights,noiseAwareFFWeights,confidenseScoreWeights = model(point_cloud_batch)
            loss = criterion([UpSamplingBlockWeights,seedGenWwights,noiseAwareFFWeights,confidenseScoreWeights], point_cloud_batchGroundTruth, point_cloud_batch)#groundTruth (10000,10000,3)
            loss.backward()
            
            optimizer.step()

            running_loss += loss.item()
            print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_dataset_radar)}], Loss: {running_loss / 10:.4f}')
            if batch_idx % 10 == 9:  
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_dataset_radar)}], Loss: {running_loss / 10:.4f}')
                running_loss = 0.0
    
    # Plotting Loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    # plt.show()
    plt.savefig("./Training and Validation Loss over Epochs.png")

    # Plotting Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy over Epochs')
    plt.legend()
    # plt.show()
    plt.savefig("./Training and Validation Accuracy over Epochs.png")
