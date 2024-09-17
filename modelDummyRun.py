from model.upSamplingBlock import UpSamplingBlock
from model.customLossFunction import CombinedLoss
from model.dataloader import PointCloudDataset

from torch.utils.data import Dataset, DataLoader

import torch


batch_size = 32
N = 1000  
input_channels = 3
output_channels = 128
device = torch.device("cpu")

total_frameStacked = torch.randn(10000,N, input_channels)
groundTruth = torch.randn(10000,300000, input_channels)

tensorDataset = PointCloudDataset(total_frameStacked)
dataloader = DataLoader(tensorDataset, batch_size=batch_size, shuffle=True)

tensorDatasetGroundTruth = PointCloudDataset(groundTruth)
dataloaderGrounfTruth = DataLoader(tensorDatasetGroundTruth, batch_size=batch_size, shuffle=True)


model = UpSamplingBlock().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = CombinedLoss(alpha=0.5).to(device)
epochs = 5

def resize_tensor_random(input_tensor, target_size=2000):
    batch_size, x, features = input_tensor.shape
    if x > target_size:
        indices = torch.randperm(x)[:target_size]
        output_tensor = input_tensor[:, indices, :]
    elif x < target_size:
        indices = torch.randint(0, x, (target_size,))
        output_tensor = input_tensor[:, indices, :]
    else:
        output_tensor = input_tensor
    return output_tensor


for epoch in range(epochs):
    running_loss = 0.0
    for batch_idx, (point_cloud_batch,point_cloud_batchGroundTruth) in enumerate(zip(dataloader, dataloaderGrounfTruth)):
        print(f"Batch {batch_idx+1}:")
        print(f"Shape of point cloud batch: {point_cloud_batch.shape}")  #(32, 1000, 3)
        print(f"Shape of GroundTruth point cloud batch: {point_cloud_batchGroundTruth.shape}")  #(32, 1000, 3)

        point_cloud_batchGroundTruth = resize_tensor_random(point_cloud_batchGroundTruth)

        print(f"Shape of GroundTruth point cloud batch Random Samp: {point_cloud_batchGroundTruth.shape}")
        optimizer.zero_grad()
        
        UpSamplingBlockWeights,seedGenWwights,noiseAwareFFWeights,confidenseScoreWeights = model(point_cloud_batch)
        loss = criterion([UpSamplingBlockWeights,seedGenWwights,noiseAwareFFWeights,confidenseScoreWeights], point_cloud_batchGroundTruth, point_cloud_batch)#groundTruth (10000,10000,3)
        # print(loss.shape)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {running_loss / 10:.4f}')
        
        
    
        # if batch_idx % 10 == 9:  
        #     print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {running_loss / 10:.4f}')
        #     running_loss = 0.0