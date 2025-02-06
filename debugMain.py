import os

# import sys
# sys.path.append(os.path.abspath("/app/utils"))

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
from torch.optim.lr_scheduler import StepLR

import numpy as np
from model.dataloader import PointCloudDataset
from model.upSamplingBlock import UpSamplingBlock
from model.customLossFunction import CombinedLoss
from sklearn.model_selection import train_test_split
import open3d as o3d
import math
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import time
from scipy.io import savemat
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import concurrent.futures
from utils.helper import *
import gc
import time
from tqdm import tqdm
from tqdm.notebook import tqdm
tqdm.pandas()


import torch
import torch.optim as optim

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch_geometric.transforms as T
import numpy as np
import os
from datetime import datetime
import shutil

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from chamfer_distance import ChamferDistance
from Emd.emd_module import emdFunction
from MMNet_V1 import Generator
from DatasetDrone_All import DatasetDrone

#defining the folder to strore the for every session 
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
processedDataFolder_name = f"./processedData/{timestamp}/"
depVis = processedDataFolder_name + "visualization/depth/"
radVis = processedDataFolder_name + "visualization/radar/"
merRadDepVis = processedDataFolder_name + "visualization/RadarDepth/"
merRadDepVisdownSample = processedDataFolder_name + "visualization/downSampledRadarDepth/"
# Check if the folder exists, if not, create it
if not os.path.exists(processedDataFolder_name):
    os.makedirs(processedDataFolder_name)
    os.makedirs(depVis)
    os.makedirs(radVis)
    os.makedirs(merRadDepVis)
    os.makedirs(merRadDepVisdownSample)
    print(f"Folder '{processedDataFolder_name}' created.")
else:
    print(f"Folder '{processedDataFolder_name}' already exists.")




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
    # print("Voxel size:",  voxel_size)
    # Step 2: Perform voxel grid downsampling
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size*0.001)
    
    # Step 3: Check if we have more points than required (voxel grid downsampling might give slightly more)
    num_downsampled_points = len(downsampled_pcd.points)
    
    if num_downsampled_points > target_num_points:
        # Randomly select 'target_num_points' from the downsampled point cloud
        indices = np.random.choice(num_downsampled_points, target_num_points, replace=False)
        downsampled_pcd = downsampled_pcd.select_by_index(indices)
    return downsampled_pcd


def pointcloud_openradar(file_name):
    print(f"{file_name} initialized")
    info_dict = get_info(file_name)
    # print_info(info_dict)
    run_data_read_only_sensor(info_dict)
    bin_filename = './datasets/radar_data/only_sensor_' + info_dict['filename'][0]
    # make fixedPoint True get fixed number of points
    pcd_data, time = generate_pcd_time(bin_filename, info_dict,fixedPoint=True,fixedPointVal=1000)
    # print(pcd_data.shape)
    return pcd_data, time


def process_bin_file(file, radarFilePath):
    """Process a single bin file and generate a CSV."""
    binFileFrame = []
    binFilePath = radarFilePath + file
    gen, timestamps = pointcloud_openradar(file_name=binFilePath)

    for pointcloud in gen:
        binFileFrame.append(pointcloud[:, :3])  # Sliced first 3 as x, y, z

    # Create a DataFrame for this bin file
    df = pd.DataFrame()
    df["datetime"] = timestamps[:gen.shape[0]]
    df["radarPCD"] = binFileFrame
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H_%M_%S.%f')

    # Save CSV
    saveCsv = radarFilePath + "csv_file/" + file.split(".")[0] + ".csv"
    radarCSVDir = radarFilePath + "csv_file/"
    if not os.path.exists(radarCSVDir):
        os.makedirs(radarCSVDir)
    
    # df.to_csv(saveCsv, index=False)
    return df  # Returning DataFrame for appending later if needed


datasetsFolderPath = './datasets/'
radarFilePath = os.path.join(datasetsFolderPath,"radar_data/")
depthFilePath = os.path.join(datasetsFolderPath,"depth_data/")
filteredBinFile = [f for f in os.listdir(radarFilePath) if os.path.isfile(os.path.join(radarFilePath, f)) and f.endswith('.bin') and not f.startswith('only_sensor')]
filteredPKLFile = [f for f in os.listdir(depthFilePath) if os.path.isfile(os.path.join(depthFilePath, f)) and f.endswith('.pkl') and not f.startswith('only_sensor')]

if len(filteredBinFile) != len(filteredPKLFile):
    print("List of CSV and BIN file is mismatched")
print("List of CSV and BIN file matched")


total_framePCD = []
total_frameRadarDF = pd.DataFrame(columns=["datetime","radarPCD"])
with concurrent.futures.ProcessPoolExecutor(max_workers= 5) as executor:
    results = list(executor.map(process_bin_file, filteredBinFile, [radarFilePath] * len(filteredBinFile)))

for df in results:
    total_frameRadarDF = pd.concat([total_frameRadarDF, df], ignore_index=True)#total_frameRadarDF.append(df, ignore_index=True)

print("BIN file Processing completed.")

#camera part
total_frameDepth = pd.DataFrame(columns=["datetime","depthPCD"])
totalDepthFrame = []
totalDepthFrameTimestamps = []
with concurrent.futures.ProcessPoolExecutor(max_workers=5) as file_executor:
        results = list(file_executor.map(load_and_process_file, filteredPKLFile))



for fileDepthFrame, fileDepthFrameTimestamps in results:
    print("Time Stamp: ",fileDepthFrameTimestamps)
    totalDepthFrame += fileDepthFrame
    totalDepthFrameTimestamps += fileDepthFrameTimestamps

print("Processing completed.")

total_frameDepth["depthPCD"] = totalDepthFrame
total_frameDepth["datetime"] = totalDepthFrameTimestamps
del totalDepthFrame 
total_frameDepth['datetime'] = pd.to_datetime(total_frameDepth['datetime'], format='%Y-%m-%d %H:%M:%S.%f')
total_frameDepth.dropna()
total_frameDepth.to_csv(processedDataFolder_name + "total_frameDepth.csv",index=False)
del totalDepthFrameTimestamps 


print("total_frameStackedRadar.shape: ",np.stack(total_frameRadarDF["radarPCD"]).shape)
# total_frameStackedDepth = np.stack(total_frameDepth["depthPCD"])
print("total_frameStackedDepth.shape: ",np.stack(total_frameDepth["depthPCD"]).shape)

# mergerdPcdDepth = pd.merge_asof(total_frameRadarDF, total_frameDepth, on='datetime',tolerance=pd.Timedelta('600ms'), direction='nearest')
# print("mergerdPcdDepth.shape: ",mergerdPcdDepth.shape)



total_frameRadarDF = total_frameRadarDF.sort_values(by='datetime', ascending=True)
total_frameDepth = total_frameDepth.sort_values(by='datetime', ascending=True)
    


mergerdPcdDepth = pd.merge_asof(total_frameRadarDF, total_frameDepth, on='datetime',tolerance=pd.Timedelta('3000ms'), direction='nearest')
print("mergerdPcdDepth.shape: ",mergerdPcdDepth.shape)
mergerdPcdDepth = mergerdPcdDepth.dropna(subset=['depthPCD'])
print("3000ms - mergerdPcdDepth after dropna.shape: ",mergerdPcdDepth.shape)


mergerdPcdDepth.to_csv(processedDataFolder_name + "/mergedRadarDepth.dat", sep = ' ', index=False)
print("mergedRadarDepth.dat Exported")

mergerdPcdDepth.to_csv(processedDataFolder_name + "/mergedRadarDepth.csv", index=False)

mergerdPcdDepth.to_pickle(processedDataFolder_name + "/mergedRadarDepth.pkl")
print("mergedRadarDepth.pkl Exported")

total_frameRadarDF.to_pickle(processedDataFolder_name + "/total_frameRadarDF.pkl")
print("total_frameRadarDF.pkl Exported")

total_frameDepth.to_pickle(processedDataFolder_name + "/total_frameDepth.pkl")
print("total_frameDepth.pkl Exported")


print(f"Data Processing Done and data exporte to file {processedDataFolder_name}")

startProcessingForNewData = True
randomDownSample = False
doDownSampling = True
doDownSampling = True
visulization = True
target_num_points = 3072

if visulization:
    for index, row in tqdm(mergerdPcdDepth.iterrows(), total=len(mergerdPcdDepth), desc="Processing frames"):
        sns.set(style="whitegrid")
        fig1 = plt.figure(figsize=(12,7))
        ax1 = fig1.add_subplot(111,projection='3d')
        img1 = ax1.scatter(mergerdPcdDepth["radarPCD"][index][:,0], mergerdPcdDepth["radarPCD"][index][:,1], mergerdPcdDepth["radarPCD"][index][:,2], cmap="jet",marker='o')
        fig1.colorbar(img1)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        # ax1.set_xlim(-10, 10)
        # ax1.set_ylim(-10, 10)
        # ax1.set_zlim(-10, 10)
        frameTime = mergerdPcdDepth["datetime"][index]
        ax1.set_title(f"Radar Point Cloud Frame {index} Time {frameTime}")
        plt.savefig(radVis + "radar_"+ str(index)+".png")
        # plt.show()
        plt.close()
    for index, row in tqdm(mergerdPcdDepth.iterrows(), total=len(mergerdPcdDepth), desc="Processing frames"):
        sns.set(style="whitegrid")
        fig2 = plt.figure(figsize=(12,7))
        ax2 = fig2.add_subplot(111,projection='3d')
        img2 = ax2.scatter(mergerdPcdDepth["depthPCD"][index][:,0], mergerdPcdDepth["depthPCD"][index][:,1],mergerdPcdDepth["depthPCD"][index][:,2], cmap="viridis",marker='o')
        fig2.colorbar(img2)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        # ax2.set_xlim(-10, 10)
        # ax2.set_ylim(-10, 10)
        # ax2.set_zlim(-10, 10)
        frameTime = mergerdPcdDepth["datetime"][index]
        ax2.set_title(f"Depth Point Cloud Frame {index} Time {frameTime}")
        plt.savefig(depVis + "depthFrame_"+ str(index)+".png")
        # plt.show()
        plt.close()

print("Importing Saved Data")
# pointcloudRadarDepth = pd.read_pickle("./mergedRadarDepth.pkl")
pointcloudRadarDepth = mergerdPcdDepth
pointcloudRadarDepth.reset_index(drop=True, inplace=True)

if doDownSampling:

    pointcloudRadarDepth["sampleDepth"] = None
    if randomDownSample:
        downsample_size = 2000  # Specify the desired downsampled size
        # downsampled_pcd = np.empty((pointcloudRadarDepth.shape[0], downsample_size, 3))
        for index, row in tqdm(pointcloudRadarDepth.iterrows(), total=pointcloudRadarDepth.shape[0], desc="Processing Rows"):            # print(index)
            indices = np.random.choice(307200, downsample_size, replace=False)  # Random indices
            # pointcloudRadarDepth.loc[index, "sampleDepth"] = pointcloudRadarDepth["depthPCD"].iloc[index][indices].tolist()  # Select points using the random indices
            pointcloudRadarDepth.at[index, "sampleDepth"] = pointcloudRadarDepth["depthPCD"].iloc[index][indices]
    else:
        downsampled_frames = []
        for index, row in tqdm(pointcloudRadarDepth.iterrows(), total=pointcloudRadarDepth.shape[0], desc="Processing Rows"):
            pcd = o3d.geometry.PointCloud()
            # print(index)
            voxel_size = 0.05  # Adjust voxel size for desired resolution
            pcd.points = o3d.utility.Vector3dVector(pointcloudRadarDepth["depthPCD"][index])
            #density based downsampling
            downsampled_pcd = density_based_downsampling(pcd, target_num_points,voxelSize=0.05)
            downsampled_points = np.asarray(downsampled_pcd.points)
            # print("downsampled_points.shape",downsampled_points.shape)
            pointcloudRadarDepth.at[index, "sampleDepth"] = downsampled_points
    pointcloudRadarDepth.to_pickle(processedDataFolder_name + "pointcloudRadarDepth.pkl")
    print("Down Samling Done, pointcloudRadarDepth.pkl Exported")
else:
    pointcloudRadarDepth = pd.read_pickle(processedDataFolder_name + "pointcloudRadarDepth.pkl")
    print("Existing Down Sampled file imported")


if False:
    # for index,row in pointcloudRadarDepth.iterrows():
    for index, row in tqdm(pointcloudRadarDepth.iterrows(), total=len(pointcloudRadarDepth), desc="Processing frames"):
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
        plt.savefig(processedDataFolder_name  + f"visualization/downSampledRadarDepth/radarDepth_{target_num_points}_{str(frameIDX)}.png")
        # plt.show()
        plt.close()
        if index ==3:
            break
    print("Sample Visulization Saved")


pkl_file = processedDataFolder_name + "mergedRadarDepth.pkl" 

outputDirTrain = processedDataFolder_name + "droneData_Train/processedData/"  
outputDirTest = processedDataFolder_name + "droneData_Test/processedData/"  

txt_file_train = processedDataFolder_name + "droneData_Train/datalist.txt" 
txt_file_test = processedDataFolder_name + "droneData_Test/datalist.txt" 

os.makedirs(outputDirTrain, exist_ok=True)
os.makedirs(outputDirTest, exist_ok=True)

df = pd.read_pickle(pkl_file)
df.reset_index(drop=True, inplace=True)

if not all(col in df.columns for col in ['radarPCD', 'depthPCD', 'datetime']):
    raise ValueError("PKL file must contain 'radarPCD', 'depthPCD', and 'datetime' columns.")


# Split data into train (80%) and test (20%)
train_size = int(0.8 * len(df))  # Adjust percentage as needed
indices = np.arange(len(df))
np.random.shuffle(indices)

train_indices = indices[:train_size]
test_indices = indices[train_size:]

with open(txt_file_train, "w") as train_out, open(txt_file_test, "w") as test_out:
    for idx in tqdm(train_indices, desc="Saving Train Data", total=len(train_indices)):
        row = df.iloc[idx]
        mat_file_name = f"{idx + 1}_mmwave.mat"
        mat_file_path = os.path.join(outputDirTrain, mat_file_name)

        savemat(mat_file_path, {
            'radarPCD': row['radarPCD'],
            'depthPCD': row['depthPCD'],
            'datetime': row['datetime']
        })
        train_out.write(mat_file_path + "\n")

    for idx in tqdm(test_indices, desc="Saving Test Data", total=len(test_indices)):
        row = df.iloc[idx]
        mat_file_name = f"{idx + 1}_mmwave.mat"
        mat_file_path = os.path.join(outputDirTest, mat_file_name)

        savemat(mat_file_path, {
            'radarPCD': row['radarPCD'],
            'depthPCD': row['depthPCD'],
            'datetime': row['datetime']
        })
        test_out.write(mat_file_path + "\n")

print(f"Exported {len(train_indices)} train .mat files to '{outputDirTrain}' and recorded in '{txt_file_train}'.")
print(f"Exported {len(test_indices)} test .mat files to '{outputDirTest}' and recorded in '{txt_file_test}'.")

