import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class PointCloudDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data: A numpy array of shape (num_samples, 1000, 3)
                  Each sample contains 1000 points with 3 coordinates (x, y, z).
        """
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        point_cloud = self.data[idx]
        point_cloud = torch.tensor(point_cloud, dtype=torch.float32)  # Convert to PyTorch tensor
        return point_cloud
num_samples = 10000  
point_cloud_data = np.random.rand(num_samples, 1000, 3) 
print(point_cloud_data.shape)
dataset = PointCloudDataset(point_cloud_data)

batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Iterate through the DataLoader and process the point cloud data
for batch_idx, point_cloud_batch in enumerate(dataloader):
    print(f"Batch {batch_idx+1}:")
    print(f"Shape of point cloud batch: {point_cloud_batch.shape}")  #(32, 1000, 3)
