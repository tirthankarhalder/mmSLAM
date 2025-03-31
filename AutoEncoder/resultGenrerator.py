import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), "..", "/app")))
os.chdir("/app")
import pandas as pd
from datetime import datetime

import cv2

import torch
import torch.nn as nn
import numpy as np
from MMNet_V1 import Generator
from DatasetDrone import DatasetDrone
from chamfer_distance import ChamferDistance
from scipy.io import savemat, loadmat
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
#import config as cfg
from Emd.emd_module import emdFunction
from datetime import datetime
from tqdm import tqdm
import os
import scipy.io
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
import sys

def save_txt(path,pred_pcd):
    '''
    pred_pcd: N by 3
    '''
    np.savetxt(path + '.txt', pred_pcd, fmt='%.6f')
    
def emd(p1,p2):
    emdist, _ = emdFunction.apply(p1, p2, 0.01, 500)
    return torch.sqrt(emdist).mean()

if __name__ == "__main__":
    # ans = input("Did you change the filename ? (yes/no) ").strip().lower()
    # if ans=='no':
    #     print("Exiting the script...")
    #     sys.exit(0)
    processedDataFolder_name = "./processedData/2025-02-11_22-23-15/"
    model_path = processedDataFolder_name + 'dronetrained/checkpoints/2025-02-11T19:41:01.537741/MMNet_ChBest.pt'#'./trained/MMNet_ChBest.pt'
    resultMatFolderPath = processedDataFolder_name + "outputDroneAll/"

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

    mergedRadarDepth = processedDataFolder_name + "mergedRadarDepth.pkl" 
    mergedRadarDepth = pd.read_pickle(mergedRadarDepth)
    mergedRadarDepth.reset_index(drop=True, inplace=True)


    rgbCsvDF = rgbCsvDF.sort_values(by='datetime', ascending=True)
    mergedRadarDepth = mergedRadarDepth.sort_values(by='datetime', ascending=True)
    
    mergedRadarDepthRgb = pd.merge_asof(mergedRadarDepth, rgbCsvDF, on='datetime',tolerance=pd.Timedelta('100ms'), direction='nearest')#change ms
    print("mergedRadarDepthRgb.shape: ",mergedRadarDepthRgb.shape)
    mergedRadarDepthRgb =mergedRadarDepthRgb.dropna(subset=['rgbFilename'])
    print("100ms - mergedRadarDepthRgb after dropna.shape: ",mergedRadarDepthRgb.shape)

    mergedRadarDepthRgb.to_csv(processedDataFolder_name + "mergedRadarDepthRgb.csv", index=False)
    mergedRadarDepthRgb.to_pickle(processedDataFolder_name + "mergedRadarDepthRgb.pkl")
    print("mergedRadarDepthRgb.pkl Exported")

    #exporting the pkl for test on slidesr
    outputDirAll = processedDataFolder_name + "droneData_All/processedData/"  
    txt_file_All = processedDataFolder_name + "droneData_All/datalist.txt" 

    os.makedirs(outputDirAll, exist_ok=True)
    if True:
        with open(txt_file_All, "w") as all_out:
            i=0
            for idx,row in tqdm(mergedRadarDepthRgb.iterrows(), desc="Saving All Data", total=len(mergedRadarDepthRgb)):
                # row = mergedPcdDepthRgb.iloc[idx]
                # mat_file_name = f"{idx + 1}_mmwave.mat"

                # #naming the matfile with repective image name
                # timestampStr, fTimestampStr = mergedRadarDepthRgb["rgbFilename"][idx].split(".")[:-1]
                # matName = f"{timestampStr}.{fTimestampStr}"

                matName = mergedRadarDepthRgb['datetime'][idx]



                mat_file_name = f"{matName}_mmwave.mat"
                mat_file_path = os.path.join(outputDirAll, mat_file_name)

                savemat(mat_file_path, {
                    'radarPCD': mergedRadarDepthRgb['radarPCD'][idx],
                    'depthPCD': mergedRadarDepthRgb['depthPCD'][idx],
                    'datetime': mergedRadarDepthRgb['datetime'][idx]
                })
                all_out.write(mat_file_path + "\n")
        print(f"Exported {len(mergedRadarDepthRgb)} testing .mat files to '{outputDirAll}' and recorded in '{txt_file_All}'.")
    
    with open(txt_file_All, "r") as f:
        mat_file_paths = [line.strip() for line in f.readlines() if line.strip().endswith(".mat")]

    mat_filenames = [path.split("/")[-1] for path in mat_file_paths]
    mat_filenames_array = np.array(mat_filenames)
    # print(len(mat_filenames_array))
    if True:
        test_dataset = DatasetDrone(processedDataFolder_name + 'droneData_All', split='test')
        print(len(test_dataset))
        test_data_loader = DataLoader(test_dataset, batch_size=1, follow_batch=['y', 'x'],shuffle=False,drop_last=False)
        print(len(test_data_loader))
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        
        G = Generator().to(device)
        
        ChD = ChamferDistance()  # chamfer loss for 

        checkpoint = torch.load(model_path)
        G.load_state_dict(checkpoint['Gen_state_dict'])

        
        # current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        timeDir = processedDataFolder_name.split("/")[-2]
        #enable it when multiple test scenios required
        # folder_path = os.path.join(resultMatFolderPath, timeDir)
        folder_path = resultMatFolderPath
        os.makedirs(folder_path,exist_ok=True)

        G.eval()
        step = 0
        # print ('Valid: ')
        loss_g =0
        each_chd = []
        each_emd = []

        progressBar = tqdm(test_data_loader, desc="Testing Progress")

        for data in progressBar:
            # print("data shape: ", len(data))
            file_name = mat_filenames_array[step]
            progressBar.set_postfix(file=file_name)
            data =data.to(device)
            # 1. Test G 
            gt_pcd = data.y     # 10000 by 3
            #x_fft = data.fft   #N:9000 M:256 H:4  W:3
            x_pos = data.x   #N:9000 M:3
            #x_img = data.imgs   #N:  M: 1   H: 80 W:80
            #x_ang = data.ang      #N:9000 M:3
            x_ini = data.ini
            batch_size = torch.max(data.y_batch)+1
            # print(x_ini.shape,x_pos.shape,gt_pcd.shape) #torch.Size([1, 3]) torch.Size([1024, 3]) torch.Size([2048, 3])
            score,init,pred = G(x_ini,x_pos,data.x_batch)
            dist1, dist2, idx1, idx2 = ChD(pred, gt_pcd.view(batch_size,-1,3))  # test G 

            g_error = 0.5*(torch.mean(torch.sqrt(dist1))) + 0.5*(torch.mean(torch.sqrt(dist2)))
            #print(g_error.size())
            loss_g += g_error.item()
            emd_error = emd(pred,gt_pcd.view(batch_size,-1,3))
            gen_data = {
            'input': x_pos.cpu().numpy().reshape((-1,3)),
            'pred_pcd': pred.detach().cpu().numpy().reshape((-1,3)),
            'gt_pcd': gt_pcd.cpu().numpy().reshape((-1,3)),
            'Chd':g_error.item(),
            'EMD':emd_error.item(),
            }
            each_chd.append(g_error.item())
            each_emd.append(emd_error.item())

            savemat(folder_path + f"/{mat_filenames_array[step]}", gen_data)
            step = step + 1
        print(loss_g/len(test_dataset))
        save_txt(folder_path + "/chd_loss.txt",np.array(each_chd))
        save_txt(folder_path + "/emd_loss.txt",np.array(each_emd))

    #plotting the graph
    all_files = os.listdir(resultMatFolderPath)
    folder_path = processedDataFolder_name + "visualization/testResultAll"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")


    mat_files = [file for file in all_files if file.endswith('.mat')]
    progressBar = tqdm(mat_files, desc="Plotting .mat files")
    # sys.exit(0)
    predictedPcd = []
    
    for mat_file in progressBar:
        progressBar.set_postfix(file=mat_file)
        file_path = os.path.join(resultMatFolderPath, mat_file)
        mat_data = scipy.io.loadmat(file_path)
        # print(f"Loaded {mat_file}")
        # print("Keys in the .mat file:", mat_data.keys())
        # for columns in header:
            # data = mat_data[columns]
            # print(f"Shape of the {columns}:", data.shape)

        header = ['input', 'pred_pcd', 'gt_pcd', 'Chd', 'EMD']

        input_points = mat_data['input']
        pred_points = mat_data['pred_pcd']
        gt_points = mat_data['gt_pcd']

        predictedPcd.append(pred_points)

        fig = plt.figure(figsize=(20, 7))
        fig.suptitle(f"Point Cloud Visualization: {mat_file}", fontsize=7, fontweight='bold')  # Main title

        ax1 = fig.add_subplot(141, projection='3d')
        scatter1 = ax1.scatter(input_points[:, 0], input_points[:, 1], input_points[:, 2], 
                            c=input_points[:, 2], cmap='viridis', s=1)
        ax1.set_title("Input Point Cloud")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")

        ax2 = fig.add_subplot(142, projection='3d')
        scatter2 = ax2.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], 
                            c=pred_points[:, 2], cmap='viridis', s=1)
        ax2.set_title("Predicted Point Cloud")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")
        ax3 = fig.add_subplot(143, projection='3d')
        scatter3 = ax3.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2], 
                            c=gt_points[:, 2], cmap='viridis', s=1)
        ax3.set_title("Ground Truth Point Cloud")
        ax3.set_xlabel("X")
        ax3.set_ylabel("Y")
        ax3.set_zlabel("Z")

        ax4 = fig.add_subplot(144)


        timestampStr = mat_file.split("_")[0]

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
        plt.savefig(f"{folder_path}/{mat_file}.png", dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close(fig)
    mergedRadarDepthRgb["predPCD"] = predictedPcd
    mergedRadarDepthRgb.to_csv(processedDataFolder_name + "mergedRadarDepthRgbPred.csv", index=False)
    mergedRadarDepthRgb.to_pickle(processedDataFolder_name + "mergedRadarDepthRgbPred.pkl")
    print("mergedRadarDepthRgbPred.pkl Exported")

        #uncommet for 3 plot 
        # fig = plt.figure(figsize=(20, 7))
        # fig.suptitle(f"Point Cloud Visualization: {mat_file}", fontsize=16, fontweight='bold')  # Main title

        # ax1 = fig.add_subplot(131, projection='3d')
        # scatter1 = ax1.scatter(input_points[:, 0], input_points[:, 1], input_points[:, 2], 
        #                     c=input_points[:, 2], cmap='viridis', s=1)
        # ax1.set_title("Input Point Cloud")
        # ax1.set_xlabel("X")
        # ax1.set_ylabel("Y")
        # ax1.set_zlabel("Z")

        # ax2 = fig.add_subplot(132, projection='3d')
        # scatter2 = ax2.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], 
        #                     c=pred_points[:, 2], cmap='viridis', s=1)
        # ax2.set_title("Predicted Point Cloud")
        # ax2.set_xlabel("X")
        # ax2.set_ylabel("Y")
        # ax2.set_zlabel("Z")
        # ax3 = fig.add_subplot(133, projection='3d')
        # scatter3 = ax3.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2], 
        #                     c=gt_points[:, 2], cmap='viridis', s=1)
        # ax3.set_title("Ground Truth Point Cloud")
        # ax3.set_xlabel("X")
        # ax3.set_ylabel("Y")
        # ax3.set_zlabel("Z")

        # plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
        # plt.savefig(f"{folder_path}/{mat_file}.png", dpi=300, bbox_inches='tight')
        # # plt.show()
        # plt.close(fig)
    







