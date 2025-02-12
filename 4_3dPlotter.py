import os
import scipy.io
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

processedDataFolder_name = "./processedData/2025-02-11_22-23-15/"
matfolder_path = processedDataFolder_name + "outputDroneTest"
all_files = os.listdir(matfolder_path)

# current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# current_datetime = matfolder_path.split("/")[-1]
# parent_dir = processedDataFolder_name + "testResult"
# folder_path = os.path.join(parent_dir, current_datetime)
folder_path = processedDataFolder_name + "visualization/testResult"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created.")
else:
    print(f"Folder '{folder_path}' already exists.")

mat_files = [file for file in all_files if file.endswith('.mat')]
for mat_file in tqdm(mat_files, desc="Plotting .mat files"):
    file_path = os.path.join(matfolder_path, mat_file)
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
    fig = plt.figure(figsize=(20, 7))
    fig.suptitle(f"Point Cloud Visualization: {mat_file}", fontsize=16, fontweight='bold')  # Main title

    ax1 = fig.add_subplot(131, projection='3d')
    scatter1 = ax1.scatter(input_points[:, 0], input_points[:, 1], input_points[:, 2], 
                        c=input_points[:, 2], cmap='viridis', s=1)
    ax1.set_title("Input Point Cloud")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    ax2 = fig.add_subplot(132, projection='3d')
    scatter2 = ax2.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], 
                        c=pred_points[:, 2], cmap='viridis', s=1)
    ax2.set_title("Predicted Point Cloud")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax3 = fig.add_subplot(133, projection='3d')
    scatter3 = ax3.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2], 
                        c=gt_points[:, 2], cmap='viridis', s=1)
    ax3.set_title("Ground Truth Point Cloud")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")
    # plt.tight_layout()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.savefig(f"{folder_path}/combined_point_clouds_drone_trained_{file_path.split('/')[-1].split('.')[0]}.png", dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close(fig)
