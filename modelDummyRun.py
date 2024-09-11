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
groundTruth = torch.randn(10000,N, input_channels)

tensorDataset = PointCloudDataset(total_frameStacked)
dataloader = DataLoader(tensorDataset, batch_size=batch_size, shuffle=True)

tensorDatasetGroundTruth = PointCloudDataset(total_frameStacked)
dataloaderGrounfTruth = DataLoader(tensorDataset, batch_size=batch_size, shuffle=True)


model = UpSamplingBlock().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = CombinedLoss(alpha=0.5).to(device)
epochs = 5


for epoch in range(epochs):
    running_loss = 0.0
    for batch_idx, (point_cloud_batch,point_cloud_batchGroundTruth) in enumerate(zip(dataloader, dataloaderGrounfTruth)):
        print(f"Batch {batch_idx+1}:")
        print(f"Shape of point cloud batch: {point_cloud_batch.shape}")  #(32, 1000, 3)
        print(f"Shape of GroundTruth point cloud batch: {point_cloud_batchGroundTruth.shape}")  #(32, 1000, 3)

        optimizer.zero_grad()

        UpSamplingBlockWeights,seedGenWwights,noiseAwareFFWeights,confidenseScoreWeights = model(point_cloud_batch)
        loss = criterion([UpSamplingBlockWeights,seedGenWwights,noiseAwareFFWeights,confidenseScoreWeights], point_cloud_batchGroundTruth)#groundTruth (10000,10000,3)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 10 == 9:  
            print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {running_loss / 10:.4f}')
            running_loss = 0.0