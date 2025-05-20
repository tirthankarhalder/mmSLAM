import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
from datetime import datetime
import pandas as pd


timestamp = "2025-05-19_15-52-07"
print("timestamp: ", timestamp)
processedDataFolder_name = f"./processedData/{timestamp}/"
depVis = processedDataFolder_name + "visualization/depth/"
radVis = processedDataFolder_name + "visualization/radar/"
merRadDepVis = processedDataFolder_name + "visualization/RadarDepth/"
merRadDepVisdownSample = processedDataFolder_name + "visualization/downSampledRadarDepth"
datasetsFolderPath = './datasets/'


if __name__ == "__main__":

    mergerdPcdDepth = pd.read_pickle(processedDataFolder_name + "mergedRadarDepth.pkl")
    mergerdPcdDepth = mergerdPcdDepth.reset_index(drop=True)
    startProcessingForNewData = True
    randomDownSample = False
    doDownSampling = True
    doDownSampling = True
    visulization = True
    target_num_points = 3072

    if True:
        startFrame = 270
        endFrame = 350
        for index, row in tqdm(mergerdPcdDepth.iterrows(), total=len(mergerdPcdDepth), desc="Processing frames"):
            if index >= startFrame:
                sns.set(style="whitegrid")
                fig1 = plt.figure(figsize=(12,7))
                ax1 = fig1.add_subplot(111,projection='3d')
                img1 = ax1.scatter(mergerdPcdDepth["radarPCD"][index][0], mergerdPcdDepth["radarPCD"][index][1], mergerdPcdDepth["radarPCD"][index][2], cmap="jet",marker='o', s=2)
                fig1.colorbar(img1)
                ax1.set_xlabel('X')
                ax1.set_ylabel('Y')
                ax1.set_zlabel('Z')
                # ax1.set_xlim(-10, 10)
                # ax1.set_ylim(-10, 10)
                # ax1.set_zlim(-10, 10)
                frameTime = mergerdPcdDepth["datetime"][index]
                ax1.set_title(f"Radar Point Cloud Frame {index} Time {frameTime}")
                plt.savefig(radVis + f"{index}_radar_"+ str(frameTime)+".png")
                # plt.show()
                plt.close()
            if index >= endFrame:
                break
            
                
        for index, row in tqdm(mergerdPcdDepth.iterrows(), total=len(mergerdPcdDepth), desc="Processing frames"):
            if index >= startFrame:
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
                plt.savefig(depVis + f"{index}_depthFrame_"+ str(frameTime)+".png")
                # plt.show()
                plt.close()
            if index == endFrame:
                break

    print("Importing Saved Data")
    # pointcloudRadarDepth = pd.read_pickle("./mergedRadarDepth.pkl")
    pointcloudRadarDepth = mergerdPcdDepth
    pointcloudRadarDepth.reset_index(drop=True, inplace=True)

    if False:
        #downsampling of DEPTH pcd if required
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
        # pointcloudRadarDepth = pd.read_pickle(processedDataFolder_name + "pointcloudRadarDepth.pkl")
        print("Existing Down Sampled file imported")


    if False:
        #plot the cpmaprison of downsampled depth pcd with actual and radar pcd
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


   
    if True:
        rgbCsvDF = pd.read_pickle(processedDataFolder_name + "rgbImage.pkl")
        mergedRadarDepth = pointcloudRadarDepth
        rgbCsvDF = rgbCsvDF.sort_values(by='datetime', ascending=True)
        mergedRadarDepth = mergedRadarDepth.sort_values(by='datetime', ascending=True)
        
        mergedRadarDepthRgb = pd.merge_asof(mergedRadarDepth, rgbCsvDF, on='datetime',tolerance=pd.Timedelta('100ms'), direction='nearest')#change ms
        print("mergedRadarDepthRgb.shape: ",mergedRadarDepthRgb.shape)
        mergedRadarDepthRgb =mergedRadarDepthRgb.dropna(subset=['rgbFilename'])
        print("100ms - mergedRadarDepthRgb after dropna.shape: ",mergedRadarDepthRgb.shape)

        mergedRadarDepthRgb.to_csv(processedDataFolder_name + "mergedRadarDepthRgb.csv", index=False)
        mergedRadarDepthRgb.to_pickle(processedDataFolder_name + "mergedRadarDepthRgb.pkl")
        print("mergedRadarDepthRgb.pkl Exported")
        for index, row in tqdm(mergedRadarDepthRgb.iterrows(), total=len(mergedRadarDepthRgb), desc="Processing frames"):            
            fig = plt.figure(figsize=(20, 7))
            fig.suptitle(f"Point Cloud Visualization: {index}", fontsize=7, fontweight='bold')  # Main title

            ax1 = fig.add_subplot(131, projection='3d')
            scatter1 = ax1.scatter(row["radarPCD"][:, 0], row["radarPCD"][:, 1], row["radarPCD"][:, 2], 
                                c=row["radarPCD"][:, 2], cmap='viridis', s=1)
            ax1.set_title("Input Point Cloud")
            ax1.set_xlabel("X")
            ax1.set_ylabel("Y")
            ax1.set_zlabel("Z")

            # ax2 = fig.add_subplot(142, projection='3d')
            # scatter2 = ax2.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], 
            #                     c=pred_points[:, 2], cmap='viridis', s=1)
            # ax2.set_title("Predicted Point Cloud")
            # ax2.set_xlabel("X")
            # ax2.set_ylabel("Y")
            # ax2.set_zlabel("Z")
            ax3 = fig.add_subplot(132, projection='3d')
            scatter3 = ax3.scatter(row["depthPCD"][:, 0], row["depthPCD"][:, 1], row["depthPCD"][:, 2], 
                                c=row["depthPCD"][:, 2], cmap='viridis', s=1)
            ax3.set_title("Ground Truth Point Cloud")
            ax3.set_xlabel("X")
            ax3.set_ylabel("Y")
            ax3.set_zlabel("Z")

            ax4 = fig.add_subplot(144)


            timestampStr = mergedRadarDepthRgb['datetime'][index]

            # timestampStr = datetime.strptime(timestampStr, "%Y-%m-%d %H:%M:%S.%f")
            # rgbFileNa = timestampStr.strftime("%Y-%m-%d %H:%M:%S.%f") + ".jpg"
            # print(rgbFileNa)

            rgbFilePt = mergedRadarDepthRgb.loc[mergedRadarDepthRgb['datetime'] == timestampStr, 'rgbFilepath']
            # print(f"{mat_file} {timestampStr} {rgbFilePt}" )
            if not rgbFilePt.empty:
                rgbFilePt = rgbFilePt.iloc[0] 
            else:
                rgbFilePt = None
            
            img = Image.open(rgbFilePt)

            ax4.imshow(img)  
            ax4.set_title("RGB Image")
            ax4.axis("off")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
            plt.savefig(f"{folder_path}/{index}.png", dpi=300, bbox_inches='tight')
            # plt.show()
            plt.close(fig)


    if False:
        
        # to generate mat file for traning and testing

        pkl_file = processedDataFolder_name + "mergedRadarDepth.pkl" 

        outputDirTrain = processedDataFolder_name + "droneData_Train/processedData/"  
        outputDirTest = processedDataFolder_name + "droneData_Test/processedData/"  

        txt_file_train = processedDataFolder_name + "droneData_Train/datalist.txt" 
        txt_file_test = processedDataFolder_name + "droneData_Test/datalist.txt" 

        os.makedirs(outputDirTrain, exist_ok=True)
        os.makedirs(outputDirTest, exist_ok=True)

        df = pd.read_pickle(pkl_file)
        df.reset_index(drop=True, inplace=True)

        if not all(col in df.columns for col in ['radarPCD', 'depthPCD', 'datetime','power']):
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
                    'datetime': row['datetime'],
                    'power': row['power']
                })
                train_out.write(mat_file_path + "\n")

            for idx in tqdm(test_indices, desc="Saving Test Data", total=len(test_indices)):
                row = df.iloc[idx]
                mat_file_name = f"{idx + 1}_mmwave.mat"
                mat_file_path = os.path.join(outputDirTest, mat_file_name)

                savemat(mat_file_path, {
                    'radarPCD': row['radarPCD'],
                    'depthPCD': row['depthPCD'],
                    'datetime': row['datetime'],
                    'power': row['power']
                })
                test_out.write(mat_file_path + "\n")

        print(f"Exported {len(train_indices)} train .mat files to '{outputDirTrain}' and recorded in '{txt_file_train}'.")
        print(f"Exported {len(test_indices)} test .mat files to '{outputDirTest}' and recorded in '{txt_file_test}'.")