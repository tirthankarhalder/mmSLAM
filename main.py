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
import open3d as o3d

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
def density_based_downsampling(pcd, target_num_points,voxelSize):
    """
    Perform density-based downsampling using voxel grid filtering.

    Args:
        pcd (open3d.geometry.PointCloud): Input point cloud.
        voxel_size (float): Size of the voxel grid. Smaller size means higher resolution.
    
    Returns:
        downsampled_pcd (open3d.geometry.PointCloud): Downsampled point cloud.
    """
     # Step 1: Estimate the diagonal length of the bounding box for the point cloud
    bbox_min = pcd.get_min_bound()
    bbox_max = pcd.get_max_bound()
    diagonal_length = np.linalg.norm(bbox_max - bbox_min)  # Calculate the length of the diagonal

    # Step 2: Calculate the downsampling ratio and estimate a scalar voxel size
    num_original_points = np.asarray(pcd.points).shape[0]
    ratio = num_original_points / target_num_points
    voxel_size = diagonal_length / (ratio ** (1/3)) 
    print("Voxel size:",  voxel_size)
    # Step 2: Perform voxel grid downsampling
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size*0.001)
    
    # Step 3: Check if we have more points than required (voxel grid downsampling might give slightly more)
    num_downsampled_points = len(downsampled_pcd.points)
    
    if num_downsampled_points > target_num_points:
        # Randomly select 'target_num_points' from the downsampled point cloud
        indices = np.random.choice(num_downsampled_points, target_num_points, replace=False)
        downsampled_pcd = downsampled_pcd.select_by_index(indices)
    return downsampled_pcd

startProcessingForNewData = False
randomDownSample = False
doDownSampling = False
visulization = False
target_num_points = 3000

if __name__ == "__main__":


    if startProcessingForNewData:
        print("Start Processing New Data")
        pointcloudRadarDepth = preprocess_ExportData(visualization=False)
    else:
        print("Importing Saved Data")
        pointcloudRadarDepth = pd.read_pickle("./mergedRadarDepth.pkl")

    pointcloudRadarDepth.reset_index(drop=True, inplace=True)

    if doDownSampling:

        pointcloudRadarDepth["sampleDepth"] = None
        if randomDownSample:
            downsample_size = 2000  # Specify the desired downsampled size
            # downsampled_pcd = np.empty((pointcloudRadarDepth.shape[0], downsample_size, 3))
            for index,row in pointcloudRadarDepth.iterrows():
                # print(index)
                indices = np.random.choice(307200, downsample_size, replace=False)  # Random indices
                # pointcloudRadarDepth.loc[index, "sampleDepth"] = pointcloudRadarDepth["depthPCD"].iloc[index][indices].tolist()  # Select points using the random indices
                pointcloudRadarDepth.at[index, "sampleDepth"] = pointcloudRadarDepth["depthPCD"].iloc[index][indices]
        else:
            downsampled_frames = []
            for index,row in pointcloudRadarDepth.iterrows():
                pcd = o3d.geometry.PointCloud()
                print(index)
                voxel_size = 0.05  # Adjust voxel size for desired resolution
                pcd.points = o3d.utility.Vector3dVector(pointcloudRadarDepth["depthPCD"][index])
                #density based downsampling
                

                downsampled_pcd = density_based_downsampling(pcd, target_num_points,voxelSize=0.05)
                downsampled_points = np.asarray(downsampled_pcd.points)
                print("downsampled_points.shape",downsampled_points.shape)
                pointcloudRadarDepth.at[index, "sampleDepth"] = downsampled_points
        pointcloudRadarDepth.to_pickle("./pointcloudRadarDepth.pkl")
        print("Down Samling Done, pointcloudRadarDepth.pkl Exported")
    else:
        pointcloudRadarDepth = pd.read_pickle("./pointcloudRadarDepth.pkl")
        print("Existing Down Sampling file importeds")
    if visulization :
        for index,row in pointcloudRadarDepth.iterrows():
            frameIDX = np.random.randint(0, pointcloudRadarDepth.shape[0])
            distancesRadar = np.linalg.norm(pointcloudRadarDepth["radarPCD"][frameIDX], axis=1)
            normalized_distances = (distancesRadar - distancesRadar.min()) / (distancesRadar.max() - distancesRadar.min())
            sns.set(style="whitegrid")
            fig = plt.figure(figsize=(12,6))
            ax1 = fig.add_subplot(131,projection='3d')
            distancesRadar = np.linalg.norm(pointcloudRadarDepth["radarPCD"][frameIDX], axis=1)
            normalized_distancesRadar = (distancesRadar - distancesRadar.min()) / (distancesRadar.max() - distancesRadar.min())
            img1 = ax1.scatter(pointcloudRadarDepth["radarPCD"][frameIDX][:, 0], pointcloudRadarDepth["radarPCD"][frameIDX][:, 1], pointcloudRadarDepth["radarPCD"][frameIDX][:, 2], c=normalized_distancesRadar,cmap = 'viridis', marker='o')
            fig.colorbar(img1)
            ax1.set_title('Radar PCD')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')

            ax2 = fig.add_subplot(132,projection='3d')
            distancesDepth = np.linalg.norm(pointcloudRadarDepth["depthPCD"][frameIDX], axis=1)
            normalized_distancesDepth = (distancesDepth - distancesDepth.min()) / (distancesDepth.max() - distancesDepth.min())
            img2 = ax2.scatter(pointcloudRadarDepth["depthPCD"][frameIDX][:, 0], pointcloudRadarDepth["depthPCD"][frameIDX][:, 1], pointcloudRadarDepth["depthPCD"][frameIDX][:, 2], c=normalized_distancesDepth, cmap = 'viridis',marker='o')
            fig.colorbar(img2)
            ax2.set_title('Depth Camera PCD')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')

            ax3 = fig.add_subplot(133,projection='3d')
            distancesSampleDepth = np.linalg.norm(pointcloudRadarDepth["sampleDepth"][frameIDX], axis=1)
            normalized_distancesSampleDepth = (distancesSampleDepth - distancesSampleDepth.min()) / (distancesSampleDepth.max() - distancesSampleDepth.min())
            img3 = ax3.scatter(pointcloudRadarDepth["sampleDepth"][frameIDX][:, 0], pointcloudRadarDepth["sampleDepth"][frameIDX][:, 1], pointcloudRadarDepth["sampleDepth"][frameIDX][:, 2], c=normalized_distancesSampleDepth, cmap = 'viridis',marker='o')
            fig.colorbar(img3)
            ax3.set_title('DownSampled Depth PCD')
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.set_zlabel('Z')
            plt.tight_layout()
            plt.savefig(f"./visualization/RadarDepth/radarDepth_{target_num_points}_{str(frameIDX)}.png")
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

            # running_loss += loss.item()
            print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_dataset_radar)}], Loss: {running_loss / 10:.4f}')
            if batch_idx % 10 == 9:  
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_dataset_radar)}], Loss: {running_loss / 10:.4f}')
                running_loss = 0.0
            # break
        # break
    
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
