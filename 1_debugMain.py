import os

# import sys
# sys.path.append(os.path.abspath("/app/utils"))

from utils.helper import *
from utils.dataProcess import *
from datetime import datetime,timedelta

import open3d as o3d
from scipy.io import savemat
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import concurrent.futures
from utils.helper import *
from tqdm import tqdm
from tqdm.notebook import tqdm
tqdm.pandas()

from tqdm import tqdm

import numpy as np
import os
from datetime import datetime
import shutil
import stat




def remove_readonly(func, path, _):
    """Change the permission of a file and retry removal."""
    os.chmod(path, stat.S_IWRITE)  
    func(path)

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


def pointcloud_time(file_name):
    print(f"{file_name} initialized")
    info_dict = get_info(file_name)
    # print_info(info_dict)
    run_data_read_only_sensor(file_name,info_dict)
    bin_filename = './datasets/radar_data/only_sensor_' + info_dict['filename'][0]
    # make fixedPoint True get fixed number of points
    pcd_data, time= generate_pcd_time(bin_filename, info_dict,fixedPoint=True,fixedPointVal=1000)
    # print(pcd_data.shape)
    return pcd_data, time

def process_bin_file(file, radarFilePath):
    """Process a single bin file and generate a CSV."""
    binFileFrame = []
    binFilesnr = []
    binFilerange = []
    binFileangle = []
    binFilepower = []
    binFiledoppler = []
    binFilePath = radarFilePath + file
    gen, timestamps= pointcloud_time(file_name=binFilePath)

    for pointcloud in gen:
        binFileFrame.append(pointcloud[:3])  # Sliced first 3 as x, y, z
        binFiledoppler.append(pointcloud[3])
        binFilesnr.append(pointcloud[4])
        binFilerange.append(pointcloud[5])
        binFileangle.append(pointcloud[6])
        binFilepower.append(pointcloud[7])

    # Create a DataFrame for this bin file
    df = pd.DataFrame()
    df["datetime"] = timestamps[:gen.shape[0]]
    df["radarPCD"] = binFileFrame
    df["doppler"] = binFiledoppler
    df["snr"] = binFilesnr
    df["range"] = binFilerange
    df["angle"] = binFileangle
    df["power"] = binFilepower
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H_%M_%S.%f')

    # Save CSV
    saveCsv = radarFilePath + "csv_file/" + file.split(".")[0] + ".csv"
    radarCSVDir = radarFilePath + "csv_file/"
    if not os.path.exists(radarCSVDir):
        os.makedirs(radarCSVDir)
    
    # df.to_csv(saveCsv, index=False)
    return df  # Returning DataFrame for appending later if needed

def pointcloud_time_openradar(file_name):
    print(f"{file_name} initialized")
    info_dict = get_info(file_name)
    # print_info(info_dict)
    run_data_read_only_sensor(file_name,info_dict)
    bin_filename = './datasets/radar_data/only_sensor_' + info_dict['filename'][0]
    # make fixedPoint True get fixed number of points
    pcd_data, time= generate_pcd_time_openradar(bin_filename, info_dict,fixedPoint=True,fixedPointVal=1000)
    # print(pcd_data.shape)
    return pcd_data, time


def process_bin_file_openradar(file, radarFilePath):
    
    """Process a single bin file and generate a CSV. This specifically design for to use openradar packages"""
    binFileFrame = []
    binFilesnr = []
    binFilerange = []
    binFileangle = []
    binFilepower = []
    binFiledoppler = []
    binFilePath = radarFilePath + file
    gen, timestamps= pointcloud_time_openradar(file_name=binFilePath)

    # Create a DataFrame for this bin file
    df = pd.DataFrame()
    df["datetime"] = timestamps[:len(gen)]
    df["radarPCD"] = gen
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H_%M_%S.%f')

    # Save CSV
    saveCsv = radarFilePath + "csv_file/" + file.split(".")[0] + ".csv"
    radarCSVDir = radarFilePath + "csv_file/"
    if not os.path.exists(radarCSVDir):
        os.makedirs(radarCSVDir)
    
    # df.to_csv(saveCsv, index=False)
    return df  # Returning DataFrame for appending later if needed

if __name__ == "__main__":
    try:
        #defining the folder to strore the for every session 
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print("timestamp: ", timestamp)
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
            results = list(executor.map(process_bin_file_openradar, filteredBinFile, [radarFilePath] * len(filteredBinFile)))

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
            # print("Time Stamp: ",fileDepthFrameTimestamps)
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

        print("total_frameRadarDF.shape",total_frameRadarDF.shape, "total_frameDepth.shape", total_frameDepth.shape)

        # print("total_frameStackedRadar.shape: ",np.stack(total_frameRadarDF["radarPCD"]).shape)
        # total_frameStackedDepth = np.stack(total_frameDepth["depthPCD"])
        # print("total_frameStackedDepth.shape: ",np.stack(total_frameDepth["depthPCD"]).shape)

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
        if True:
            import cv2
            from PIL import Image
            folder_path = processedDataFolder_name + "visualization/testResultAll"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                print(f"Folder '{folder_path}' created.")
            else:
                print(f"Folder '{folder_path}' already exists.")
            #to genrate the comaprison visulization depth and rgb
            datasetFolder = "./datasets/image_data/"
            image_data = []
            for subdir, _, files in os.walk(datasetFolder):
                for file in files:
                    if file.endswith(".jpg"):  
                        file_path = os.path.join(subdir, file)
                        renamedFile = file[:-4]
                        renamedFiletimestamp = datetime.strptime(renamedFile, "%Y-%m-%d_%H_%M_%S_%f")
                        renamedFileFormattedTime = renamedFiletimestamp.strftime("%Y-%m-%d %H:%M:%S.%f") + ".jpg"
                        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
                        date_str, time_hr,time_min,time_sec, microseconds = file[:-4].split("_")
                        datetime_str = f"{date_str} {time_hr}:{time_min}:{time_sec}.{microseconds}"
                        timestamp = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S.%f")
                        formatted_timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")
                        image_data.append([formatted_timestamp, image, renamedFileFormattedTime, file_path])

            rgbCsvDF = pd.DataFrame(image_data, columns=[ "datetime", "rgbImage","rgbFilename", "rgbFilepath"])
            rgbCsvDF["datetime"] = pd.to_datetime(rgbCsvDF["datetime"], format="%Y-%m-%d %H:%M:%S.%f")
            rgbCsvPath = os.path.join(processedDataFolder_name, "rgbImage.csv")
            rgbCsvDF.to_csv(rgbCsvPath, index=False)
            rgbCsvDF.to_pickle(processedDataFolder_name + "rgbImage.pkl")
            print(f"pkl file saved at: {rgbCsvPath}")

        
    except Exception as e:
        print("An unexpected error occurred:", e)
    finally:
        ans = input("Do you want keep the data? (yes/no)")
        if ans == "no":
            try:
                subprocess.run(["rm", "-rf", processedDataFolder_name], check=True)
                print(f"Directory '{processedDataFolder_name}' removed successfully.")
            except subprocess.CalledProcessError as e:
                print(f"Error removing directory: {e}")