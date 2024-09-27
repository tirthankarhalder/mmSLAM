import os
from helper import *
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

def pointcloud_openradar(file_name):
    info_dict = get_info(file_name)
    run_data_read_only_sensor(info_dict)
    bin_filename = './datasets/radar_data/only_sensor_' + info_dict['filename'][0]
    pcd_data, time = generate_pcd_time(bin_filename, info_dict,fixedPoint=True)
    # print(pcd_data.shape)
    return pcd_data, time


if __name__ == "__main__":

    visualization = False
                
    datasetsFolderPath = './datasets/'
    radarFilePath = os.path.join(datasetsFolderPath,"radar_data/")
    depthFilePath = os.path.join(datasetsFolderPath,"depth_data/")
    filteredBinFile = [f for f in os.listdir(radarFilePath) if os.path.isfile(os.path.join(radarFilePath, f)) and f.endswith('.bin') and not f.startswith('only_sensor')]
    filteredCsvFile = [f for f in os.listdir(depthFilePath) if os.path.isfile(os.path.join(depthFilePath, f)) and f.endswith('.csv') and not f.startswith('only_sensor')]

    total_framePCD = []
    total_frameRadar = pd.DataFrame(columns=["datetime","pcd"])
    for file in filteredBinFile:#interate over all bin and stack
        binFilePath = radarFilePath+file
        gen,timestamps=pointcloud_openradar(file_name =binFilePath)
        frameID = 1
        for pointcloud in gen:
            if frameID ==1 and visualization:
                sns.set(style="whitegrid")
                fig = plt.figure(figsize=(12,7))
                ax = fig.add_subplot(111,projection='3d')
                img = ax.scatter(pointcloud[:,0], pointcloud[:,1], pointcloud[:,2], cmap="jet",marker='o')
                fig.colorbar(img)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')

                plt.savefig("./visualization/radar/OpenRadarFixedPointPCD.png")
                # plt.show()
                plt.close()
            total_framePCD.append(pointcloud[:,:3])#sliced 1st 3 as x y z
            frameID+=1
        df = pd.DataFrame()
        df["datetime"] = timestamps
        df["pcd"] = total_framePCD
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H_%M_%S.%f')
        saveCsv = radarFilePath + "csv_file/" + file.split(".")[0] + ".csv"
        radarCSVDir = radarFilePath + "csv_file/"
        if not os.path.exists(saveCsv):
            os.makedirs(radarCSVDir)
        total_frameRadar.append(df)
        df.to_csv(saveCsv)#this will crate indivisula csv file fr each bin file

    total_frameStacked = np.stack(total_framePCD)
    print("total_frameStacked.shape: ",total_frameStacked.shape)


    # total_frameDepth = pd.DataFrame()
    # for file in filteredCsvFile:#iterate over all csv and stack as df
    #     csvFilePath = depthFilePath+file
    #     depthDf = pd.read_csv(csvFilePath)

    csvFilePath = depthFilePath + '*.csv'
    dfs = [pd.read_csv(file) for file in glob.glob(csvFilePath)]
    total_frameDepth = pd.concat(dfs, ignore_index=True)#merger all csv

    for index, row in total_frameDepth.iterrows():
        for i in range(len(row["x"])):
            pass
        #complete tommorrow make frame from x y z
    total_frameDepth['datetime'] = pd.to_datetime(total_frameDepth['datetime'], format='%Y-%m-%d %H:%M:%S.%f')
    total_frameDepth.dropna()
    total_frameDepth.to_csv("total_frameDepth.csv")



    # #check new samples radar data plot with pcd column
    # sns.set(style="whitegrid")
    # fig = plt.figure(figsize=(12,7))
    # ax = fig.add_subplot(111,projection='3d')
    # for index,row in total_frameDepth.iterrows():
    #     img = ax.scatter(total_framePCDDf["pcd"][0][:,0], total_framePCDDf["pcd"][0][:,1],total_framePCDDf["pcd"][0][:,2], cmap="viridis",marker='o')
    #     fig.colorbar(img)
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z')

    #     plt.savefig("./visualization/radar/radar_"+ str(index)+".png")
    #     # plt.show()
    #     plt.close()
    #     if index == 0:
    #         break
    if visualization:
        # check new samples depth data plot with pcdDepth column
        sns.set(style="whitegrid")
        fig = plt.figure(figsize=(12,7))
        ax = fig.add_subplot(111,projection='3d')
        for index,row in total_frameDepth.iterrows():
            
            lisx = np.array(ast.literal_eval(total_frameDepth["depthPCD"][0]))[:,0]
            lisy = np.array(ast.literal_eval(total_frameDepth["depthPCD"][0]))[:,1]
            lisz = np.array(ast.literal_eval(total_frameDepth["depthPCD"][0]))[:,2]
            # print(li[0])
            img = ax.scatter(lisx, lisy,lisz, cmap="viridis",marker='o')
            fig.colorbar(img)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            plt.savefig("./visualization/depth/depthFrame_"+ str(index)+".png")
            # plt.show()
            plt.close()
            if index == 0:
                break

    if visualization:
        # check new samples depth data plot with individual x y z column
        sns.set(style="whitegrid")
        fig = plt.figure(figsize=(12,7))
        ax = fig.add_subplot(111,projection='3d')
        for index,row in total_frameDepth.iterrows():
            
            lisx = ast.literal_eval(row["x"])
            lisy = ast.literal_eval(row["y"])
            lisz = ast.literal_eval(row["z"])
            # print(li[0])
            img = ax.scatter(lisx, lisy,lisz, cmap="viridis",marker='o')
            fig.colorbar(img)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            plt.savefig("./visualization/depth/depthXYZ_"+ str(index)+".png")
            # plt.show()
            plt.close()
            if index == 0:
                break


    #totalframePCDDf and total_frameDepth is final merged file 

    mergerdPcdDepth = pd.merge_asof(total_frameRadar, total_frameDepth, on='datetime',tolerance=pd.Timedelta('2us'), direction='nearest')

    print("mergerdPcdDepth.shape: ",mergerdPcdDepth.shape)


    for index,row in mergerdPcdDepth.iterrows():
        pass








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
