import os
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
def save_txt(path,pred_pcd):
    '''
    pred_pcd: N by 3
    '''
    np.savetxt(path + '.txt', pred_pcd, fmt='%.6f')
    
def emd(p1,p2):
    emdist, _ = emdFunction.apply(p1, p2, 0.01, 500)
    return torch.sqrt(emdist).mean()

if __name__ == "__main__":
    processedDataFolder_name = "./processedData/2025-02-01_14-28-44/"
    datasetFolder = "./datasets/image_data/"
    image_data = []
    for subdir, _, files in os.walk(datasetFolder):
        for file in files:
            if file.endswith(".jpg"):  
                file_path = os.path.join(subdir, file)
                image = cv2.imread(file_path, cv2.IMREAD_COLOR)
                date_str, time_hr,time_min,time_sec, microseconds = file[:-4].split("_")
                datetime_str = f"{date_str} {time_hr}_{time_min}_{time_sec}.{microseconds}"
                timestamp = datetime.strptime(datetime_str, "%Y-%m-%d %H_%M_%S.%f")
                formatted_timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S.000")
                # print(formatted_timestamp)
                image_data.append([formatted_timestamp,image, file, file_path])

    rgbCsvDF = pd.DataFrame(image_data, columns=[ "datetime", "rgbImage","filename", "filepath"])
    rgbCsvDF["datetime"] = pd.to_datetime(rgbCsvDF["datetime"], format="%Y-%m-%d %H:%M:%S.%f")

    mergedRadarDepth = processedDataFolder_name + "mergedRadarDepth.pkl" 
    mergedRadarDepth = pd.read_pickle(mergedRadarDepth)
    mergedRadarDepth.reset_index(drop=True, inplace=True)


    rgbCsvDF = rgbCsvDF.sort_values(by='datetime', ascending=True)
    mergedRadarDepth = mergedRadarDepth.sort_values(by='datetime', ascending=True)
    
    mergerdPcdDepthRgb = pd.merge_asof(mergedRadarDepth, rgbCsvDF, on='datetime',tolerance=pd.Timedelta('5ms'), direction='nearest')
    print("mergerdPcdDepthRgb.shape: ",mergerdPcdDepthRgb.shape)
    mergerdPcdDepthRgb =mergerdPcdDepthRgb.dropna(subset=['filename'])
    print("5ms - mergerdPcdDepth after dropna.shape: ",mergerdPcdDepthRgb.shape)


    mergerdPcdDepthRgb.to_csv(processedDataFolder_name + "mergerdPcdDepthRgb.csv", index=False)
    mergerdPcdDepthRgb.to_pickle(processedDataFolder_name + "mergerdPcdDepthRgb.pkl")
    print("mergerdPcdDepthRgb.pkl Exported")

    #exporting the pkl for test on slides
    outputDirAll = processedDataFolder_name + "droneData_All/processedData/"  
    txt_file_All = processedDataFolder_name + "droneData_All/datalist.txt" 

    os.makedirs(outputDirAll, exist_ok=True)

    with open(txt_file_All, "w") as all_out:
        for idx in tqdm(len(mergedRadarDepth), desc="Saving Train Data", total=len(mergedRadarDepth)):
            row = mergedRadarDepth.iloc[idx]
            mat_file_name = f"{idx + 1}_mmwave.mat"
            mat_file_path = os.path.join(outputDirAll, mat_file_name)

            savemat(mat_file_path, {
                'radarPCD': row['radarPCD'],
                'depthPCD': row['depthPCD'],
                'datetime': row['datetime']
            })
            all_out.write(mat_file_path + "\n")
    print(f"Exported {len(mergedRadarDepth)} testing .mat files to '{outputDirAll}' and recorded in '{txt_file_train}'.")


    test_dataset = DatasetDrone(processedDataFolder_name + 'droneData_All', split='test')
    
    test_data_loader = DataLoader(test_dataset, batch_size=1, follow_batch=['y', 'x'],shuffle=False,drop_last=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    G = Generator().to(device)
    
    ChD = ChamferDistance()  # chamfer loss for 

    model_path = processedDataFolder_name + 'dronetrained/checkpoints/2025-02-01T14:43:14.917241/MMNet_ChBest.pt'#'./trained/MMNet_ChBest.pt'
    checkpoint = torch.load(model_path)
    G.load_state_dict(checkpoint['Gen_state_dict'])

    parentDir = processedDataFolder_name + "outputDroneAll/"
    # current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    timeDir = processedDataFolder_name.split("/")[-2]
    #enable it when multiple test scenios required
    # folder_path = os.path.join(parentDir, timeDir)
    folder_path = parentDir
    os.makedirs(folder_path,exist_ok=True)

    G.eval()
    step = 0
    print ('Valid: ')
    loss_g =0
    each_chd = []
    each_emd = []
    for data in test_data_loader:
        # print("data shape: ", len(data))
        print(f"File no: {step+1} generated")
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

        savemat(folder_path + "/result"+str(step)+".mat", gen_data)
        step = step + 1
    print(loss_g/len(test_dataset))
save_txt(folder_path + "/chd_loss.txt",np.array(each_chd))
save_txt(folder_path + "/emd_loss.txt",np.array(each_emd))





