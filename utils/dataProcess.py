import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import ast

from helper import *

def pointcloud_openradar(file_name):
    info_dict = get_info(file_name)
    run_data_read_only_sensor(info_dict)
    bin_filename = './datasets/radar_data/only_sensor_' + info_dict['filename'][0]
    pcd_data, time = generate_pcd_time(bin_filename, info_dict,fixedPoint=True)
    # print(pcd_data.shape)
    return pcd_data, time

def preprocess_ExportData(visualization=False):                
    datasetsFolderPath = './datasets/'
    radarFilePath = os.path.join(datasetsFolderPath,"radar_data/")
    depthFilePath = os.path.join(datasetsFolderPath,"depth_data/")
    filteredBinFile = [f for f in os.listdir(radarFilePath) if os.path.isfile(os.path.join(radarFilePath, f)) and f.endswith('.bin') and not f.startswith('only_sensor')]
    filteredCsvFile = [f for f in os.listdir(depthFilePath) if os.path.isfile(os.path.join(depthFilePath, f)) and f.endswith('.csv') and not f.startswith('only_sensor')]

    total_framePCD = []
    total_frameRadar = pd.DataFrame(columns=["datetime","radarPCD"])
    for file in filteredBinFile:#interate over all bin and stack
        binFilePath = radarFilePath+file
        gen,timestamps=pointcloud_openradar(file_name =binFilePath)
        frameID = 1
        for pointcloud in gen:
            total_framePCD.append(pointcloud[:,:3])#sliced 1st 3 as x y z
            frameID+=1
        df = pd.DataFrame()
        df["datetime"] = timestamps
        df["radarPCD"] = total_framePCD
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H_%M_%S.%f')
        saveCsv = radarFilePath + "csv_file/" + file.split(".")[0] + ".csv"
        radarCSVDir = radarFilePath + "csv_file/"
        if not os.path.exists(radarCSVDir):
            os.makedirs(radarCSVDir)
        total_frameRadar.append(df)
        df.to_csv(saveCsv,index=False)#this will crate indivisula csv file fr each bin file

    total_frameStacked = np.stack(total_framePCD)
    print("total_frameStacked.shape: ",total_frameStacked.shape)

    csvFilePath = depthFilePath + '*.csv'
    dfs = [pd.read_csv(file) for file in glob.glob(csvFilePath)]
    total_frameDepth = pd.concat(dfs, ignore_index=True)#merger all csv

   
    totalDepthFrame = []
    for index,row in total_frameDepth.iterrows():
        iPointer=0
        frame = []
        lisx = ast.literal_eval(row["x"])
        lisy = ast.literal_eval(row["y"])
        lisz = ast.literal_eval(row["z"])
        for i in range(len(lisx)):
            frame.append(list([lisx[i],lisy[i],lisz[i]]))
        totalDepthFrame.append(frame)
    total_frameDepth["depthPCD"] = totalDepthFrame

    total_frameDepth['datetime'] = pd.to_datetime(total_frameDepth['datetime'], format='%Y-%m-%d %H:%M:%S.%f')
    total_frameDepth.dropna()
    total_frameDepth.to_csv("total_frameDepth.csv",index=False)

    mergerdPcdDepth = pd.merge_asof(total_frameRadar, total_frameDepth, on='datetime',tolerance=pd.Timedelta('2us'), direction='nearest')

    print("mergerdPcdDepth.shape: ",mergerdPcdDepth.shape)

    
    if visualization:
        for index,row in mergerdPcdDepth.iterrows():
            sns.set(style="whitegrid")
            fig1 = plt.figure(figsize=(12,7))
            ax1 = fig1.add_subplot(111,projection='3d')
            img1 = ax1.scatter(mergerdPcdDepth["radarPCD"][:,0], mergerdPcdDepth["radarPCD"][:,1], mergerdPcdDepth["radarPCD"][:,2], cmap="jet",marker='o')
            fig1.colorbar(img)
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')

            plt.savefig("./visualization/radar/radar_"+ str(index)+".png")
            # plt.show()
            plt.close()

            sns.set(style="whitegrid")
            fig2 = plt.figure(figsize=(12,7))
            ax2 = fig2.add_subplot(111,projection='3d')
            for index,row in total_frameDepth.iterrows():
                img2 = ax2.scatter(total_frameDepth["depthPCD"][:,0], total_frameDepth["depthPCD"][:,1],total_frameDepth["depthPCD"][:,2], cmap="viridis",marker='o')
                fig2.colorbar(img)
                ax2.set_xlabel('X')
                ax2.set_ylabel('Y')
                ax2.set_zlabel('Z')

                plt.savefig("./visualization/depth/depthFrame_"+ str(index)+".png")
                # plt.show()
                plt.close()
                # if index == 0:
                #     break
    mergerdPcdDepth.to_csv("mergedRadarDepth.dat", sep = ' ', index=False)



if __name__ == "__main__":

    preprocess_ExportData(visualization=True)