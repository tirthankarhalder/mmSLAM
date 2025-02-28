import pandas as pd
import os
from tqdm import tqdm
from scipy.io import savemat, loadmat
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np


processedDataFolder_name = "./processedData/2025-02-11_22-23-15/"
model_path = processedDataFolder_name + 'dronetrained/checkpoints/2025-02-11T19:41:01.537741/MMNet_ChBest.pt'#'./trained/MMNet_ChBest.pt'
resultMatFolderPath = processedDataFolder_name + "outputDroneAll/"
all_files = os.listdir(resultMatFolderPath)
folder_path = processedDataFolder_name + "visualization/testResultPowerAll"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created.")
else:
    print(f"Folder '{folder_path}' already exists.")

mergedRadarDepthRgbPred = processedDataFolder_name + "mergedRadarDepthRgbPred.pkl" 
mergedRadarDepthRgbPred = pd.read_pickle(mergedRadarDepthRgbPred)
mergedRadarDepthRgbPred.reset_index(drop=True, inplace=True)

for index, row in tqdm(mergedRadarDepthRgbPred.iterrows(), total=len(mergedRadarDepthRgbPred), desc="Processing frames"):
    mergedRadarDepthRgbPred['datetime'] = pd.to_datetime(mergedRadarDepthRgbPred['datetime'])
    frameIDX = index
    # print(frameIDX)
    sns.set(style="whitegrid")
    fig = plt.figure(figsize=(20,7))
    fig.suptitle(f"Point Cloud Visualization: {mergedRadarDepthRgbPred['datetime'][frameIDX]}", fontsize=7, fontweight='bold')  # Main title

    ax1 = fig.add_subplot(141,projection='3d')
    power_values = np.array(mergedRadarDepthRgbPred["power"][frameIDX])  
    img1 = ax1.scatter(mergedRadarDepthRgbPred["radarPCD"][frameIDX][:, 0], mergedRadarDepthRgbPred["radarPCD"][frameIDX][:, 1], mergedRadarDepthRgbPred["radarPCD"][frameIDX][:, 2], c=mergedRadarDepthRgbPred["power"][frameIDX],cmap = 'viridis', s=1)
    fig.colorbar(img1)
    ax1.set_title('Radar PCD')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    ax2 = fig.add_subplot(142,projection='3d')
    img2 = ax2.scatter(mergedRadarDepthRgbPred["depthPCD"][frameIDX][:, 0], mergedRadarDepthRgbPred["depthPCD"][frameIDX][:, 1], mergedRadarDepthRgbPred["depthPCD"][frameIDX][:, 2], c=mergedRadarDepthRgbPred["depthPCD"][frameIDX][:, 2], cmap = 'viridis',s=1)
    fig.colorbar(img2)
    ax2.set_title('Depth Camera PCD')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    ax3 = fig.add_subplot(143,projection='3d')
    img3 = ax3.scatter(mergedRadarDepthRgbPred["predPCD"][frameIDX][:, 0], mergedRadarDepthRgbPred["predPCD"][frameIDX][:, 1], mergedRadarDepthRgbPred["predPCD"][frameIDX][:, 2], c=mergedRadarDepthRgbPred["predPCD"][frameIDX][:, 2],cmap = 'viridis',s=1)
    fig.colorbar(img3)
    ax3.set_title('Predicted PCD')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')

    ax4 = fig.add_subplot(144)
    img4 = Image.open(mergedRadarDepthRgbPred["rgbFilepath"][frameIDX])
    ax4.imshow(img4)  
    ax4.set_title("RGB Image")
    ax4.axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    figName = mergedRadarDepthRgbPred["datetime"][frameIDX]
    plt.savefig(f"{folder_path}/{figName}.png", dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close(fig)
            