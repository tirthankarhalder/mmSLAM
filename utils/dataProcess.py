import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import ast

from utils.helper import *

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
    filteredPKLFile = [f for f in os.listdir(depthFilePath) if os.path.isfile(os.path.join(depthFilePath, f)) and f.endswith('.pkl') and not f.startswith('only_sensor')]

    if len(filteredBinFile) != len(filteredPKLFile):
        return None
    print("List of CSV and BIN file matched")
    # else: check all bin and csv file same
    #     for csvFile in filteredCsvFile:
    #         csvST = "_".join(csvFile.split("_")[1:5])
    #         for binFile in filteredBinFile:
    #             binST = "_".join(binFile.split("_")[1:5])

    #radar part
    total_framePCD = []
    total_frameRadarDF = pd.DataFrame(columns=["datetime","radarPCD"])
    for file in filteredBinFile:#interate over all bin and stack
        binFileFrame = []
        binFilePath = radarFilePath+file
        gen,timestamps=pointcloud_openradar(file_name =binFilePath)
        frameID = 1
        for pointcloud in gen:
            binFileFrame.append(pointcloud[:,:3])#sliced 1st 3 as x y z
            total_framePCD.append(pointcloud[:,:3])

            frameID+=1
        df = pd.DataFrame()
        df["datetime"] = timestamps[:gen.shape[0]]
        df["radarPCD"] = binFileFrame
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H_%M_%S.%f')
        saveCsv = radarFilePath + "csv_file/" + file.split(".")[0] + ".csv"
        radarCSVDir = radarFilePath + "csv_file/"
        if not os.path.exists(radarCSVDir):
            os.makedirs(radarCSVDir)
        total_frameRadarDF = total_frameRadarDF.append(df,ignore_index=True)
        df.to_csv(saveCsv,index=False)#this will crate indivisula csv file fr each bin file
    total_frameRadarDF.to_csv("total_frameRadarDF.csv", index=False)

    total_frameStacked = np.stack(total_framePCD)
    print("total_frameStacked.shape: ",total_frameStacked.shape)


    #camera part
    total_frameDepth = pd.DataFrame(columns=["datetime","depthPCD"])
    totalDepthFrame = []
    totalDepthFrameTimestamps = []
    for file in filteredPKLFile:
        # depthPCDObjects = []
        frameFile = []
        # npPointcloud = np.load(depthFilePath +file,allow_pickle=True)
        with open(depthFilePath+file, 'rb') as f:
            # pointcloud, timestamp = pickle.load(f)
            while True:
                try:
                    depthObj = pickle.load(f)
                    frameFile+=depthObj
                    # depthPCDObjects.append(depthObj)
                except EOFError: 
                    break
                except Exception as e:
                    print(f"Error reading file: {e}")
        
            
        for i in range(len(frameFile)):
            frame = []
            x = frameFile[i][0]['f0']
            y = frameFile[i][0]['f1']
            z = frameFile[i][0]['f2']
            # time = frameFile[str(i)][1]
            # for j in range(len(x)):
            #     frame.append(np.array([x[i],y[i],z[i]]))
            totalDepthFrame.append(frameFile[i][0])
            totalDepthFrameTimestamps.append(frameFile[i][1])
            # break



    # csvFilePath = depthFilePath + '*.csv'
    # dfs = [pd.read_csv(file) for file in glob.glob(csvFilePath)]
    # total_frameDepth = pd.concat(dfs, ignore_index=True)#merger all csv

   
    # totalDepthFrame = []
    # for index,row in total_frameDepth.iterrows():
    #     iPointer=0
    #     frame = []
    #     lisx = ast.literal_eval(row["x"])
    #     lisy = ast.literal_eval(row["y"])
    #     lisz = ast.literal_eval(row["z"])
    #     for i in range(len(lisx)):
    #         frame.append(list([lisx[i],lisy[i],lisz[i]]))
    #     totalDepthFrame.append(frame)
    
    total_frameDepth["depthPCD"] = totalDepthFrame
    total_frameDepth["datetime"] = totalDepthFrameTimestamps

    total_frameDepth['datetime'] = pd.to_datetime(total_frameDepth['datetime'], format='%Y-%m-%d %H:%M:%S.%f')
    total_frameDepth.dropna()
    total_frameDepth.to_csv("total_frameDepth.csv",index=False)

    mergerdPcdDepth = pd.merge_asof(total_frameRadarDF, total_frameDepth, on='datetime',tolerance=pd.Timedelta('2us'), direction='nearest')
    
    print("mergerdPcdDepth.shape: ",mergerdPcdDepth.shape)
    mergerdPcdDepth.to_csv("mergedRadarDepth.dat", sep = ' ', index=False)
    mergerdPcdDepth.to_csv("mergedRadarDepth.csv", index=False)

    mergerdPcdDepth.to_pickle("./mergedRadarDepth.pkl")
    total_frameRadarDF.to_pickle("./total_frameRadarDF.pkl")
    total_frameDepth.to_pickle("./total_frameDepth.pkl")


    
    if visualization:
        for index,row in mergerdPcdDepth.iterrows():
            sns.set(style="whitegrid")
            fig1 = plt.figure(figsize=(12,7))
            ax1 = fig1.add_subplot(111,projection='3d')
            img1 = ax1.scatter(mergerdPcdDepth["radarPCD"][index], mergerdPcdDepth["radarPCD"][index], mergerdPcdDepth["radarPCD"][index], cmap="jet",marker='o')
            fig1.colorbar(img1)
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
            img2 = ax2.scatter(total_frameDepth["depthPCD"][index], total_frameDepth["depthPCD"][index],total_frameDepth["depthPCD"][index], cmap="viridis",marker='o')
            fig2.colorbar(img2)
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')

            plt.savefig("./visualization/depth/depthFrame_"+ str(index)+".png")
            # plt.show()
            plt.close()
            # if index == 0:
            #     break



if __name__ == "__main__":

    preprocess_ExportData(visualization=False)