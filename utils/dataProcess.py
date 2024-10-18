import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import concurrent.futures
from utils.helper import *
import gc
def pointcloud_openradar(file_name):
    info_dict = get_info(file_name)
    run_data_read_only_sensor(info_dict)
    bin_filename = './datasets/radar_data/only_sensor_' + info_dict['filename'][0]
    pcd_data, time = generate_pcd_time(bin_filename, info_dict,fixedPoint=True)
    # print(pcd_data.shape)
    return pcd_data, time

def process_frames(frameSPerFile, file):
    """Process frames in a single file and return depth frames and timestamps."""
    fileDepthFrame = []
    fileDepthFrameTimestamps = []

    for i in range(len(frameSPerFile)):
        frame = []
        x = frameSPerFile[i][0]['f0']
        y = frameSPerFile[i][0]['f1']
        z = frameSPerFile[i][0]['f2']
        print(f"Frame initialized: {i} {file}")

        # for pointIndex in range(len(x)):
        #     frame.append(np.array([x[pointIndex], y[pointIndex], z[pointIndex]]))
        frame = np.array([x, y, z]).T

        fileDepthFrame.append(frame)
        fileDepthFrameTimestamps.append(frameSPerFile[i][1])   
    return fileDepthFrame, fileDepthFrameTimestamps

def load_and_process_file(file):
    """Load data from the PKL file, process it, and return the frames and timestamps."""
    datasetsFolderPath = './datasets/'
    depthFilePath = os.path.join(datasetsFolderPath,"depth_data/")
    frameSPerFile = []
    try:
        with open(depthFilePath + file, 'rb') as f:
            while True:
                try:
                    depthObj = pickle.load(f)
                    frameSPerFile += depthObj
                except EOFError:
                    break
                except Exception as e:
                    print(f"Error reading file: {e}")
        
        # Process frames in parallel (per file)
        return process_frames(frameSPerFile, file)

    except Exception as e:
        print(f"Failed to process {file}: {e}")
        return [], []
   
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

def save_partial_results(fileDepthFrame, fileDepthFrameTimestamps, file):
    """Save partial results to disk to avoid memory overflow."""
    # Save to a CSV or append to a PKL file
    save_file = f"./datasets/depth_data/processed/processed_{file}.pkl"
    with open(save_file, 'wb') as f:
        pickle.dump((fileDepthFrame, fileDepthFrameTimestamps), f)


# def process_frames_in_chunks(frameSPerFile, file, chunk_size=100):
#     """Process frames in chunks to optimize memory usage."""
#     total_depth_frame = []
#     total_depth_timestamps = []

#     # Process in chunks
#     for start in range(0, len(frameSPerFile), chunk_size):
#         end = min(start + chunk_size, len(frameSPerFile))
#         chunk = frameSPerFile[start:end]

#         # Parallelize processing of frames in this chunk
#         with concurrent.futures.ProcessPoolExecutor() as frame_executor:
#             results = list(frame_executor.map(
#                 lambda i: process_single_frame(chunk[i], i + start, file),
#                 range(len(chunk))
#             ))

#         # Unpack the processed frames and timestamps
#         file_depth_frame, file_depth_frame_timestamps = zip(*results)
        
#         # Append processed data to the main list (or save it to disk)
#         total_depth_frame.extend(file_depth_frame)
#         total_depth_timestamps.extend(file_depth_frame_timestamps)

#         # Release memory after each chunk
#         del chunk
#         del results
#         gc.collect()  # Force garbage collection

#     return total_depth_frame, total_depth_timestamps

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

    with concurrent.futures.ThreadPoolExecutor(max_workers= 5) as executor:
        results = list(executor.map(process_bin_file, filteredBinFile, [radarFilePath] * len(filteredBinFile)))

    for df in results:
        total_frameRadarDF = total_frameRadarDF.append(df, ignore_index=True)

    print("BIN file Processing completed.")

    # total_frameStackedRadar = np.stack(total_frameRadarDF["radarPCD"])
    print("total_frameStackedRadar.shape: ",np.stack(total_frameRadarDF["radarPCD"]).shape)


    #camera part
    total_frameDepth = pd.DataFrame(columns=["datetime","depthPCD"])
    totalDepthFrame = []
    totalDepthFrameTimestamps = []

    # Parallel processing of all PKL files
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as file_executor:
        # Read and process all files in parallel
        for file, result in zip(filteredPKLFile, file_executor.map(load_and_process_file, filteredPKLFile)):
            fileDepthFrame, fileDepthFrameTimestamps = result

            # Save the processed frames and timestamps incrementally
            save_partial_results(fileDepthFrame, fileDepthFrameTimestamps, file)
            
            # Clear memory for the next file
            del fileDepthFrame, fileDepthFrameTimestamps
            gc.collect()  # Force garbage collection to free memory

            
    # with concurrent.futures.ThreadPoolExecutor(max_workers=2) as file_executor:
    #     results = list(file_executor.map(load_and_process_file, filteredPKLFile))
    # for fileDepthFrame, fileDepthFrameTimestamps in results:
    #     totalDepthFrame += fileDepthFrame
    #     totalDepthFrameTimestamps += fileDepthFrameTimestamps

    print("Processing completed.")

    total_frameDepth["depthPCD"] = totalDepthFrame
    # total_frameDepth["datetime"] = totalDepthFrameTimestamps
    del totalDepthFrame 
    total_frameDepth['datetime'] = pd.to_datetime(total_frameDepth['datetime'], format='%Y-%m-%d %H:%M:%S.%f')
    total_frameDepth.dropna()
    # total_frameDepth.to_csv("total_frameDepth.csv",index=False)
    del totalDepthFrameTimestamps 

    # total_frameStackedDepth = np.stack(total_frameDepth["depthPCD"])
    print("total_frameStackedDepth.shape: ",np.stack(total_frameDepth["depthPCD"]).shape)

    mergerdPcdDepth = pd.merge_asof(total_frameRadarDF, total_frameDepth, on='datetime',tolerance=pd.Timedelta('2us'), direction='nearest')
    print("mergerdPcdDepth.shape: ",mergerdPcdDepth.shape)

    mergerdPcdDepth.to_csv("mergedRadarDepth.dat", sep = ' ', index=False)
    print("mergedRadarDepth.dat Exported")

    mergerdPcdDepth.to_csv("mergedRadarDepth.csv", index=False)

    mergerdPcdDepth.to_pickle("./mergedRadarDepth.pkl")
    print("mergedRadarDepth.pkl Exported")

    total_frameRadarDF.to_pickle("./total_frameRadarDF.pkl")
    print("total_frameRadarDF.pkl Exported")

    # total_frameDepth.to_pickle("./total_frameDepth.pkl")
    # print("total_frameDepth.pkl Exported")


    print("Data Processing Done")
    
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

    return mergerdPcdDepth

if __name__ == "__main__":

    preprocess_ExportData(visualization=False)